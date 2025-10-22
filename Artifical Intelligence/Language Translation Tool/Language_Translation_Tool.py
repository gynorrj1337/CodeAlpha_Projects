import tkinter as tk
from tkinter import ttk, messagebox
from deep_translator import GoogleTranslator
from gtts import gTTS
import pyperclip
import os
from playsound3 import playsound

# ---------------------------------------------
# 🌍 Language Code Mapping
# ---------------------------------------------
language_code_map = {
    "Auto-detect": "auto",
    "English": "en",
    "Tamil": "ta",
    "Sinhala": "si",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese (Simplified)": "zh-CN",
    "Hindi": "hi",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Bengali": "bn",
    "Polish": "pl",
    "Turkish": "tr",
    "Swedish": "sv",
    "Dutch": "nl",
    "Danish": "da",
    "Finnish": "fi",
    "Norwegian": "no",
    "Czech": "cs",
    "Greek": "el",
    "Thai": "th",
    "Hebrew": "he",
    "Hungarian": "hu",
    "Romanian": "ro",
    "Ukrainian": "uk",
    "Malay": "ms",
    "Vietnamese": "vi",
    "Malayalam": "ml",
    "Telugu": "te",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Kannada": "kn",
    "Punjabi": "pa",
    "Serbian": "sr",
    "Albanian": "sq",
    "Croatian": "hr",
    "Bosnian": "bs",
    "Macedonian": "mk",
    "Swahili": "sw",
    "Lithuanian": "lt",
    "Latvian": "lv",
    "Estonian": "et",
    "Icelandic": "is",
    "Bulgarian": "bg",
    "Afrikaans": "af",
    "Welsh": "cy",
    "Latin": "la"
}

# ---------------------------------------------
# 🧠 Functional Logic
# ---------------------------------------------
def translate_text():
    source_text = source_text_box.get("1.0", tk.END).strip()
    source_lang = source_lang_combo.get()
    target_lang = target_lang_combo.get()

    if not source_text:
        messagebox.showwarning("Input Error", "Please enter text to translate.")
        return

    source_code = language_code_map.get(source_lang, "auto")
    target_code = language_code_map.get(target_lang, "en")

    try:
        translated = GoogleTranslator(source=source_code, target=target_code).translate(source_text)
        target_text_box.delete("1.0", tk.END)
        target_text_box.insert(tk.END, translated)
    except Exception as e:
        messagebox.showerror("Translation Error", str(e))

def speak_text():
    text = target_text_box.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("No Text", "Please translate text first.")
        return

    target_lang = target_lang_combo.get()
    tts_lang = language_code_map.get(target_lang, "en")

    try:
        tts = gTTS(text=text, lang=tts_lang)
        audio_file = "speech.mp3"
        tts.save(audio_file)
        playsound(audio_file)
        os.remove(audio_file)
    except Exception as e:
        messagebox.showerror("Speech Error", str(e))

def copy_text():
    text = target_text_box.get("1.0", tk.END).strip()
    if text:
        pyperclip.copy(text)
        messagebox.showinfo("Copied", "Translated text copied to clipboard!")
    else:
        messagebox.showwarning("No Text", "No translated text to copy!")

# ---------------------------------------------
# 🎨 Modern GUI Setup
# ---------------------------------------------
root = tk.Tk()
root.title("🌍 AI-Powered Language Translator")
root.geometry("960x600")
root.configure(bg="#f0f8ff")
root.resizable(False, False)

style = ttk.Style()
style.theme_use('clam')

# Custom colors and fonts
PRIMARY_COLOR = "#0077b6"
SECONDARY_COLOR = "#00b4d8"
BUTTON_COLOR = "#023e8a"
TEXTBOX_BG = "#ffffff"
FONT = ("Segoe UI", 11)

# Title
title = tk.Label(root, text="🌍 AI-Powered Language Translator", font=("Segoe UI", 20, "bold"),
                 bg="#f0f8ff", fg=PRIMARY_COLOR)
title.pack(pady=20)

# Frame for Text Boxes
text_frame = tk.Frame(root, bg="#f0f8ff")
text_frame.pack(pady=10)

# Source Text Box
tk.Label(text_frame, text="Enter Text", font=FONT, bg="#f0f8ff").grid(row=0, column=0, padx=10, pady=5, sticky="w")
source_text_box = tk.Text(text_frame, height=10, width=45, font=("Segoe UI", 10), bg=TEXTBOX_BG, bd=2, relief="groove")
source_text_box.grid(row=1, column=0, padx=10, pady=5)

# Translated Text Box
tk.Label(text_frame, text="Translated Text", font=FONT, bg="#f0f8ff").grid(row=0, column=1, padx=10, pady=5, sticky="w")
target_text_box = tk.Text(text_frame, height=10, width=45, font=("Segoe UI", 10), bg="#f9f9f9", bd=2, relief="groove")
target_text_box.grid(row=1, column=1, padx=10, pady=5)

# Language Selection
lang_frame = tk.Frame(root, bg="#f0f8ff")
lang_frame.pack(pady=10)

languages = list(language_code_map.keys())

tk.Label(lang_frame, text="From:", font=FONT, bg="#f0f8ff").grid(row=0, column=0, padx=10)
source_lang_combo = ttk.Combobox(lang_frame, values=languages, width=25, state="readonly", font=("Segoe UI", 10))
source_lang_combo.grid(row=0, column=1, padx=10)
source_lang_combo.set("Auto-detect")

tk.Label(lang_frame, text="To:", font=FONT, bg="#f0f8ff").grid(row=0, column=2, padx=10)
target_lang_combo = ttk.Combobox(lang_frame, values=languages, width=25, state="readonly", font=("Segoe UI", 10))
target_lang_combo.grid(row=0, column=3, padx=10)
target_lang_combo.set("English")

# Buttons
button_frame = tk.Frame(root, bg="#f0f8ff")
button_frame.pack(pady=20)

def styled_button(master, text, command, bg, fg):
    return tk.Button(master, text=text, command=command,
                     font=("Segoe UI", 11, "bold"), bg=bg, fg=fg, width=15, height=1,
                     activebackground="#ddd", relief="flat", cursor="hand2")

translate_btn = styled_button(button_frame, "🌐 Translate", translate_text, BUTTON_COLOR, "white")
translate_btn.grid(row=0, column=0, padx=10)

copy_btn = styled_button(button_frame, "📋 Copy", copy_text, "#2e8b57", "white")
copy_btn.grid(row=0, column=1, padx=10)

speak_btn = styled_button(button_frame, "🔊 Speak", speak_text, "#ff7f50", "white")
speak_btn.grid(row=0, column=2, padx=10)

# Footer
footer = tk.Label(root, text="Created with ❤️ by Gynorrj | Powered by Deep Translator & gTTS",
                  font=("Segoe UI", 9), bg="#f0f8ff", fg="#444")
footer.pack(side="bottom", pady=10)

root.mainloop()
