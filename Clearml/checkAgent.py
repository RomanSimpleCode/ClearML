import sys
from clearml import Task

if sys.platform == "win32":
    Task.ignore_requirements("pywin32")

task = Task.init(
    project_name="Tutorial",
    task_name="Agent FIX FINAL",
    reuse_last_task_id=False,
)

task.execute_remotely(queue_name="default")

print("Correct")
