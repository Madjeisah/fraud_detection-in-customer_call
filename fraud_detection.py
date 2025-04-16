import re
import torch
import whisperx
import whisper
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import rec_audio
from colorama import init, Fore,Back,Style
import warnings


warnings.filterwarnings('ignore')


#  Call our WhisperX model (An open-source) for transcribing 

# Set params
device = "cuda" 
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

# 1. Transcribe with original whisper (batched)
# model = whisperx.load_model("small.en", device, compute_type=compute_type) #"large-v2" "small.en"
# audio = whisperx.load_audio("customer_call.wav")
# result = model.transcribe(audio, batch_size=batch_size)
# print(result["segments"]) 

# Try whisper
model = whisper.load_model("small.en") #"large-v2" "small.en"
result = model.transcribe("customer_call.wav")
print(result["text"]) 


# model_name = model="facebook/bart-large-mnli"
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)


# Fraud-related keywords
fraud_keywords = set([
    "fraud", "scam", "fake", "false", "cheat", "deceptive", "bogus", "counterfeit", "urgent",
    "phony", "forged", "hoax", "shady", "suspicious", "dishonest", "unlawful", "risk", "payment",
    "unauthorized", "unauthorised", "money laundering", "identity theft", "phishing","steal", "payment information",
    "chargeback", "credit card", "credit card fraud", "wire fraud", "Ponzi scheme", "pyramid scheme", "transactions",
    "hacking", "malware", "ransomware", "keylogger", "spoofing", "spyware", "earliest convenience",
    "fake website", "data breach", "click fraud", "brute force attack", "breach", "fraudulent",
    "warranty scam", "return fraud", "fake product", "knockoff", "gift card scam", "link",
    "tax evasion", "ghost employee", "fake invoices", "insurance fraud", "password",
    "medical billing fraud", "staged accident", "investment fraud", "illegal",
    "pump and dump", "rug pull", "crypto scam", "fake ICO", "get-rich-quick scheme",
    "transaction error", "hacked", "unauthenticated", "authentication", "forgery"
])

risk_labels = ["Very Low Risk", "Low Risk", "Medium Risk", "High Risk", "Very High Risk"]


def keyword_detection(text):
    """
    Detect fraud-related keywords in customer call transcript.

    Args:
        text (str): The customer call transcript.

    Returns:
        list: List of detected fraud-related words.
    """
    # Check for keyword matches
    det_keywords = [word for word in fraud_keywords if re.search(rf'\b{re.escape(word)}\b', text, re.IGNORECASE)]
    
    return det_keywords


def bert_analysis(text):
    """
    Use BERT model to analyze the risk level of context in the fraud-related keywords.

    Args:
        text (str): The customer call transcript.

    Returns:
        dict: A dictionary of fraud probability and classification label [from "Very Low Risk" - "Very High Risk"].
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)


    fraud_pred = torch.argmax(probabilities, dim=1).item()

    return {
        "Fraud Risk Level": risk_labels[fraud_pred],
        "Confidence": f"{probabilities[0][fraud_pred].item() * 100:.2f}%"
    }

def confidence_level(keyword_counts, risk_label, sys_confidence):
    """
    Args:
        ---
    Returns:
        ---
    """
    # Determine the risk level based on the keywords counts
    if keyword_counts >= 5:
        risk_index = 4  # "Very High Risk"
    elif keyword_counts >= 3:
        risk_index = 3  # "High Risk"
    elif keyword_counts >= 2:
        risk_index = 2  # "Medium Risk"
    elif keyword_counts == 1:
        risk_index = 1  # "Low Risk"
    else:
        risk_index = 0  # "Very Low Risk"

    # Assign system comfidence based on the level risk
    if risk_label in ["High Risk", "Very High Risk"]:
        if float(sys_confidence.strip('%')) >= 80:
            risk_index = max(4, risk_index)  # Ensure "Very High Risk"
        elif float(sys_confidence.strip('%')) >= 60:
            risk_index = max(3, risk_index)  # Ensure at least "High Risk"

    return risk_labels[risk_index]


def fraud_detection_system(text):
    """

    Args:
        text (str): Customer call transcrpt

    Returns:
        dict: A dictionary of the System's risk assesment.
    """
    detect_keywords = keyword_detection(text)
    
    if detect_keywords:
        output = bert_analysis(text)
    else:
        output = {"Fraud Risk Level": "Very Low Risk", "Confidence": "99.99%"}

    risk_assesment = confidence_level(len(detect_keywords), 
                                   output["Fraud Risk Level"], 
                                   output["Confidence"])
    return{
        "Sytem Detected Fruad Keywords": detect_keywords,
        "System Analysis": output,
        "Risk Assesment Level": risk_assesment
    }


def main():
    
    # Example Transcript
    customer_call = """
    I received a call saying my bank account had unauthorized transactions. 
    They asked me for my credit card details, which seemed suspicious.
    Later, I saw fraudulent charges on my statement. This looks like a scam!
    """

    # Process text
    processed_text = " ".join([entry["text"].strip() for entry in result["segments"]])

    print(f"\n\n{Fore.LIGHTBLACK_EX}Customer call transcibing {Fore.RESET}")
    print(f"{Fore.MAGENTA}Customer: {Fore.RESET}{Back.WHITE} {processed_text}{Back.RESET}") 

    # Fraud Detection System
    fraud_result = fraud_detection_system(processed_text)

    # 📌 Display Results
    print("\n========================================")
    print("    🚨 Fraud Detection System 🚨")
    print("========================================")
    if fraud_result['Sytem Detected Fruad Keywords']:
        print(f"🔍 {Back.RED}Sytem detected suspicious information{Back.RESET}: {Fore.LIGHTYELLOW_EX}{', '.join(fraud_result['Sytem Detected Fruad Keywords'])}{Fore.RESET}")
    else:
        print(f"✅ {Back.GREEN} No suspicious information detected.{Back.RESET}")

    print(f"🤖 Risk Assesment Level: {fraud_result['Risk Assesment Level']}")
    print(f"🎯 Sytem Confidence Score: {fraud_result['System Analysis']['Confidence']}\n")


if __name__ == '__main__':
    main()
    
    
