from langchain_core.tools import tool

#create as much tool as you want as a python function
@tool
def note_tool(note):
    """
    saves a note to a local file

    Args:
        note: the text note to save
    """

    with open("notes.txt","a") as f:
        f.write(note+"\n")