import copy
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class SubtaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ObjectMemory:
    name: str
    initial_position: List[float]
    current_position: List[float]
    last_updated: float = field(default_factory=time.time)

    def update_position(self, new_pos: List[float]):
        self.current_position = list(new_pos)
        self.last_updated = time.time()


@dataclass
class SubtaskState:
    id: int
    action: str
    object: str
    target: str
    status: SubtaskStatus = SubtaskStatus.PENDING
    attempts: int = 0
    result: Optional[Dict] = None
    execution_result: Optional[Dict] = None
    verification_result: Optional[Dict] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class TaskStateManager:
    
    def __init__(self):
        self.plan: Dict = {}
        self.subtasks: List[SubtaskState] = []
        self.current_subtask_index: int = 0
        self.object_memory: Dict[str, ObjectMemory] = {}
        self.instruction: str = ""
        self.session_id: str = ""
        self.state_transitions: List[Dict] = []
        self._transition_callbacks: List[Callable] = []
    
    def initialize(self, plan: Dict, instruction: str = "", object_positions: Dict[str, List[float]] = None):
        self.plan = copy.deepcopy(plan)
        self.instruction = instruction
        self.session_id = f"session_{int(time.time())}"
        
        self.subtasks = []
        for st in self.plan.get("subtasks", []):
            subtask_state = SubtaskState(
                id=st.get("id", len(self.subtasks) + 1),
                action=st.get("action", "pick_and_place"),
                object=st.get("object", "unknown"),
                target=st.get("target", "unknown"),
                status=SubtaskStatus.PENDING,
            )
            self.subtasks.append(subtask_state)
        
        self.object_memory = {}
        for obj_name in self._get_all_objects():
            self.object_memory[obj_name] = ObjectMemory(
                name=obj_name,
                initial_position=object_positions.get(obj_name, [0, 0, 0]) if object_positions else [0, 0, 0],
                current_position=object_positions.get(obj_name, [0, 0, 0]) if object_positions else [0, 0, 0],
            )
        
        self.current_subtask_index = 0
        self._log_transition("initialized", None, None, {"subtask_count": len(self.subtasks)})
    
    def _get_all_objects(self) -> List[str]:
        objects = set()
        for st in self.subtasks:
            objects.add(st.object)
            objects.add(st.target)
        return list(objects)
    
    def has_pending_subtasks(self) -> bool:
        return self.current_subtask_index < len(self.subtasks)
    
    def get_next_subtask(self) -> Optional[Dict]:
        if not self.has_pending_subtasks():
            return None
        
        subtask = self.subtasks[self.current_subtask_index]
        self._set_status(subtask.id, SubtaskStatus.IN_PROGRESS)
        
        return {
            "id": subtask.id,
            "action": subtask.action,
            "object": subtask.object,
            "target": subtask.target,
            "attempt_number": subtask.attempts + 1,
        }
    
    def get_current_subtask(self) -> Optional[SubtaskState]:
        if self.current_subtask_index < len(self.subtasks):
            return self.subtasks[self.current_subtask_index]
        return None
    
    def start_subtask(self, subtask) -> None:
        """Mark a subtask as started (sets started_at, used for timing).

        Drivers that iterate the plan directly rather than calling
        get_next_subtask() must call this, otherwise started_at stays None and
        get_summary()["total_time"] is always 0.0.
        """
        subtask_id = subtask.get("id") if isinstance(subtask, dict) else subtask
        self._set_status(subtask_id, SubtaskStatus.IN_PROGRESS)

    def mark_complete(self, subtask: Dict, execution_result: Dict = None, verification_result: Dict = None):
        if isinstance(subtask, dict):
            subtask_id = subtask.get("id", self.current_subtask_index + 1)
        else:
            subtask_id = subtask
        
        self._set_status(subtask_id, SubtaskStatus.COMPLETED, execution_result, verification_result)
        
        self._log_transition(
            "subtask_completed",
            subtask_id,
            SubtaskStatus.COMPLETED.value,
            {"verification": verification_result}
        )
        
        self.current_subtask_index += 1
    
    def mark_failed(self, subtask: Dict, execution_result: Dict = None, verification_result: Dict = None,
                    reason: str = "Max retries exceeded"):
        if isinstance(subtask, dict):
            subtask_id = subtask.get("id", self.current_subtask_index + 1)
        else:
            subtask_id = subtask
        
        self._set_status(subtask_id, SubtaskStatus.FAILED, execution_result, verification_result, reason)
        
        self._log_transition(
            "subtask_failed",
            subtask_id,
            SubtaskStatus.FAILED.value,
            {"reason": reason}
        )
        
        self.current_subtask_index += 1
    
    def increment_retry(self, subtask: Dict) -> int:
        if isinstance(subtask, dict):
            subtask_id = subtask.get("id", self.current_subtask_index + 1)
        else:
            subtask_id = subtask
        
        for st in self.subtasks:
            if st.id == subtask_id:
                st.attempts += 1
                self._log_transition(
                    "retry",
                    subtask_id,
                    SubtaskStatus.IN_PROGRESS.value,
                    {"attempt": st.attempts}
                )
                return st.attempts
        return 0
    
    def _set_status(self, subtask_id: int, status: SubtaskStatus, execution_result: Dict = None,
                    verification_result: Dict = None, failure_reason: str = None):
        for st in self.subtasks:
            if st.id == subtask_id:
                st.status = status
                if execution_result:
                    st.execution_result = execution_result
                if verification_result:
                    st.verification_result = verification_result
                
                if status == SubtaskStatus.IN_PROGRESS:
                    st.started_at = time.time()
                elif status in [SubtaskStatus.COMPLETED, SubtaskStatus.FAILED]:
                    st.completed_at = time.time()
                break
    
    def update_object_position(self, obj_name: str, position: List[float]):
        if obj_name in self.object_memory:
            self.object_memory[obj_name].update_position(position)
            self._log_transition(
                "object_moved",
                None,
                None,
                {"object": obj_name, "new_position": position}
            )
    
    def get_object_position(self, obj_name: str) -> Optional[List[float]]:
        if obj_name in self.object_memory:
            return self.object_memory[obj_name].current_position
        return None
    
    def _log_transition(self, event_type: str, subtask_id: Optional[int], new_status: Optional[str],
                        metadata: Dict = None):
        transition = {
            "timestamp": time.time(),
            "event": event_type,
            "subtask_id": subtask_id,
            "new_status": new_status,
            "metadata": metadata or {},
        }
        self.state_transitions.append(transition)
        
        for callback in self._transition_callbacks:
            try:
                callback(transition)
            except Exception as e:
                pass
    
    def add_transition_callback(self, callback: Callable):
        self._transition_callbacks.append(callback)
    
    def get_progress(self) -> Dict:
        total = len(self.subtasks)
        completed = sum(1 for st in self.subtasks if st.status == SubtaskStatus.COMPLETED)
        failed = sum(1 for st in self.subtasks if st.status == SubtaskStatus.FAILED)
        in_progress = sum(1 for st in self.subtasks if st.status == SubtaskStatus.IN_PROGRESS)
        pending = sum(1 for st in self.subtasks if st.status == SubtaskStatus.PENDING)
        
        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "pending": pending,
            "progress_percent": (completed / total * 100) if total > 0 else 0,
            "current_index": self.current_subtask_index,
        }
    
    def get_summary(self) -> Dict:
        total_time = 0
        successful_tasks = [st for st in self.subtasks if st.status == SubtaskStatus.COMPLETED]
        failed_tasks = [st for st in self.subtasks if st.status == SubtaskStatus.FAILED]
        
        for st in self.subtasks:
            if st.started_at and st.completed_at:
                total_time += st.completed_at - st.started_at
        
        return {
            "instruction": self.instruction,
            "session_id": self.session_id,
            "total_subtasks": len(self.subtasks),
            "successful": len(successful_tasks),
            "failed": len(failed_tasks),
            "total_time": total_time,
            "subtasks": [
                {
                    "id": st.id,
                    "action": st.action,
                    "object": st.object,
                    "target": st.target,
                    "status": st.status.value,
                    "attempts": st.attempts,
                }
                for st in self.subtasks
            ],
        }
    
    def reset(self):
        self.plan = {}
        self.subtasks = []
        self.current_subtask_index = 0
        self.object_memory = {}
        self.instruction = ""
        self.session_id = ""
        self.state_transitions = []


def create_state_manager() -> TaskStateManager:
    return TaskStateManager()


if __name__ == "__main__":
    print("Testing Task State Manager...")
    
    manager = create_state_manager()
    
    test_plan = {
        "subtasks": [
            {"id": 1, "action": "pick_and_place", "object": "red_block", "target": "bin"},
            {"id": 2, "action": "stack", "object": "blue_block", "target": "green_block"},
        ]
    }
    
    object_positions = {
        "red_block": [0.25, 0.1, 0.04],
        "blue_block": [0.25, -0.1, 0.04],
        "green_block": [0.15, 0.1, 0.04],
        "bin": [0.5, 0.0, 0.04],
    }
    
    manager.initialize(test_plan, instruction="test instruction", object_positions=object_positions)
    print("\nInitialized plan:")
    print(json.dumps(manager.get_summary(), indent=2))
    
    print("\nProgress:", manager.get_progress())
    
    print("\n--- Simulating subtask 1 ---")
    subtask1 = manager.get_next_subtask()
    print(f"Got subtask: {subtask1}")
    
    print("\nProgress:", manager.get_progress())
    
    print("\n--- Simulating retry 1 ---")
    manager.increment_retry(subtask1)
    print(f"Attempt after increment: {subtask1}")
    
    print("\n--- Marking subtask 1 complete ---")
    manager.mark_complete(subtask1, execution_result={"duration": 1.2}, verification_result={"success": True})
    
    print("\nProgress:", manager.get_progress())
    print("\nState transitions:")
    for t in manager.state_transitions:
        print(f"  {t}")
    
    print("\n--- Simulating subtask 2 ---")
    subtask2 = manager.get_next_subtask()
    print(f"Got subtask: {subtask2}")
    
    print("\n--- Simulating multiple retries ---")
    for i in range(3):
        attempts = manager.increment_retry(subtask2)
        print(f"Retry {i + 1}: attempt count = {attempts}")
    
    print("\n--- Marking subtask 2 failed ---")
    manager.mark_failed(subtask2, reason="Max retries exceeded")
    
    print("\nFinal progress:", manager.get_progress())
    print("\nFinal summary:")
    print(json.dumps(manager.get_summary(), indent=2))
    
    print("\nTask State Manager test complete!")
