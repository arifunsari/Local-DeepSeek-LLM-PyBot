

### ✅ `README.md`

````markdown
# 🤖 Local DeepSeek LLM Python Assistant

This project is a **locally-hosted AI code assistant and document Q&A system** powered by **DeepSeek LLMs** using **Ollama** and **LangChain**. It provides two core features:

---

## 🚀 Features

### 1. 🧠 DeepSeek Code Companion
> Your AI pair programmer

- Chat-based Python coding assistant
- Model options: `deepseek-r1:1.5b` and `deepseek-r1:3b`
- Built-in debugging advice and solution generation
- Powered by **LangChain**, **Ollama**, and **Streamlit**

---

### 2. 📘 DocuMind AI
> An intelligent PDF reader for research or technical documents

- Upload a PDF, and ask questions about its content
- Embeds documents using `deepseek-r1:1.5b` embeddings
- Answers queries using contextual retrieval and LLM generation
- Fully local and private

---

## 🛠️ Problem This Solves

Many developers and researchers want to use AI locally for:
- Writing and debugging Python code
- Asking intelligent questions about technical PDFs

But most solutions are cloud-based and expensive. This project brings **high-quality LLM capabilities fully offline** using **Ollama**.

---

## ⚙️ Setup Instructions

### 📦 1. Install Dependencies

Make sure you have Python 3.10+ and [Ollama](https://ollama.ai/) installed locally.

```bash
pip install -r requirements.txt
````

### 💡 2. Pull the DeepSeek Models

Install the models into Ollama (this may take a few minutes):

```bash
ollama pull deepseek-coder:1.5b
ollama pull deepseek-coder:3b
```

> If you're using `deepseek-r1`, adjust commands accordingly.

---

### 🧪 3. Run the Apps

There are two Streamlit apps:

#### 1. Run Code Companion

```bash
streamlit run app.py
```

#### 2. Run Document Q\&A

```bash
streamlit run rag.py
```

---

## 📂 Folder Structure

```bash
├── app.py                    # Code companion chatbot
├── rag.py                    # Document Q&A system
├── requirements.txt
├── Document_Store/
│   └── your_pdf.pdf          # Folder to store uploaded PDFs
```

---

## 📌 Notes

* Ollama should be running locally on `http://localhost:11434`
* Ensure DeepSeek models are downloaded before running the app
* Works offline once models are downloaded
* Tested with `deepseek-r1:1.5b` and `deepseek-r1:3b`

---

## 🙌 Credits

Built with:

* [LangChain](https://www.langchain.com/)
* [Streamlit](https://streamlit.io/)
* [Ollama](https://ollama.ai/)
* [DeepSeek LLMs](https://github.com/deepseek-ai)

---

## 📜 License

MIT License


Let me know if you want me to add images, badges, or a video demo section.
