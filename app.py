
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load model from HF Hub
model_name = "neelam-builds/mistral-7b-medical-finetuned"

bnb_config = BitsAndBytesConfig(load_in_4bit=True)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
)
model.eval()

def medical_assistant(question, history):
    if not question.strip():
        return "", history
    
    inputs = tokenizer(question, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if question in response:
        response = response.replace(question, "").strip()
    
    history = history or []
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": response})
    return "", history

with gr.Blocks() as demo:
    gr.HTML("""
    <div style="text-align:center; padding:30px 0 15px 0;">
        <h1 style="font-size:2.5em; background:linear-gradient(90deg,#00d2ff,#7b2ff7);
        -webkit-background-clip:text; -webkit-text-fill-color:transparent;
        font-weight:800; margin-bottom:8px;">🏥 MediAsk</h1>
        <p style="color:#9999bb; font-size:0.95em;">
        Your personal medical knowledge assistant — ask anything, get clear answers</p>
    </div>
    """)

    chatbot = gr.Chatbot(label="", height=460, show_label=False, render_markdown=True)

    with gr.Row():
        msg = gr.Textbox(placeholder="Type your medical question here...",
                        show_label=False, scale=9, container=False)
        send = gr.Button("Send ➤", scale=1)

    gr.HTML("<p style='color:#777799; font-size:0.82em; text-align:center; margin:12px 0 6px 0'>💡 Try one of these:</p>")

    with gr.Row():
        q1 = gr.Button("🩺 Symptoms of diabetes?")
        q2 = gr.Button("❤️ What is hypertension?")
        q3 = gr.Button("🤒 What causes fever?")
        q4 = gr.Button("💊 Uses of paracetamol?")

    clear = gr.Button("🗑️ Clear Chat")

    gr.HTML("""
    <div style="text-align:center; color:#555577; font-size:0.78em; padding:20px;">
        ⚠️ MediAsk is for informational purposes only.<br>
        Always consult a qualified doctor for medical advice.
    </div>
    """)

    history = gr.State([])

    send.click(medical_assistant, [msg, history], [msg, chatbot])
    msg.submit(medical_assistant, [msg, history], [msg, chatbot])
    q1.click(lambda h: medical_assistant("What are the symptoms of diabetes?", h), [history], [msg, chatbot])
    q2.click(lambda h: medical_assistant("What is hypertension?", h), [history], [msg, chatbot])
    q3.click(lambda h: medical_assistant("What causes fever?", h), [history], [msg, chatbot])
    q4.click(lambda h: medical_assistant("What is paracetamol used for?", h), [history], [msg, chatbot])
    clear.click(lambda: ([], []), None, [chatbot, history])

demo.launch()
