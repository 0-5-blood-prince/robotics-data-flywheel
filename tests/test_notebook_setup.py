"""Exercise source activation in a running interpreter without site .pth loading."""
import json
from pathlib import Path
import subprocess
import sys
import unittest

class NotebookSetupTests(unittest.TestCase):
    def test_current_kernel_import_after_source_activation(self):
        root=Path(__file__).resolve().parents[1]
        notebook=json.loads((root/'notebooks/flywheel.ipynb').read_text())
        setup=''.join(notebook['cells'][1]['source'])
        activation=setup[setup.index('# Activate src'):]
        script='''import sys,importlib
from pathlib import Path
ROOT=Path(sys.argv[1])
try:
 import robot_flywheel
except ModuleNotFoundError:
 pass
else:
 raise AssertionError('Precondition: package unexpectedly importable')
'''+activation
        result=subprocess.run([sys.executable,'-I','-S','-c',script,str(root)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('Package ready:',result.stdout)
