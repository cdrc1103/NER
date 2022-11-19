import gradio as gr

from inference import inference_pipeline

EXAMPLE_INPUT = (
    "My name is Wolfgang and I live in Berlin. Recently, I started working "
    + "at Brainlab were I work as a data scientist."
)

TITLE = "Named Entity Recognition"
DESCRIPTION = (
        "Insert a text of your choice and let it be processed by the "
        + "NER Model for Organizations, Persons, and Locations."
)

inputs = gr.Textbox(label="Input Text")
outputs = gr.HTML(label="Annotated Text")

demo = gr.Interface(
    fn=inference_pipeline,
    inputs=inputs,
    outputs=outputs,
    title=TITLE,
    examples=[[EXAMPLE_INPUT]],
    description=DESCRIPTION,
    allow_flagging="never"
)
demo.launch()


