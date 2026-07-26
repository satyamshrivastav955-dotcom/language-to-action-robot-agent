import os, zipfile, sys

src = os.path.join(os.environ['TEMP'], 'pptxwork')
dst = sys.argv[1]
if os.path.exists(dst):
    os.remove(dst)
z = zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED)
for root, dirs, files in os.walk(src):
    for n in files:
        p = os.path.join(root, n)
        arc = os.path.relpath(p, src).replace(os.sep, '/')
        z.write(p, arc)
z.close()
print('packed ->', dst)
