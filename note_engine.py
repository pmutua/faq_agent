from llama_index.legacy.tools import FunctionTool
import os 

note_file = os.path.join("data", "notes.txt")

def save_note(note):
    if not os.path.exists(note_file):
        open(note_file, "a")
    with open(note_file, "a") as f: 
        f.writelines([note+ "\n"])

    return "note saved"


note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name="note_saver",
    description="This tool can save text based note to a file for a user"
)