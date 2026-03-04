import tiktoken
import os
from tqdm import tqdm
from openai import OpenAI
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
import argparse
import requests

BASE_URL = "https://53330c3e49d1.ngrok-free.app"
CUSTOM_API_URL = f"{BASE_URL}/v1/chat/completions"


class CustomLLMClient:
    def __init__(self, base_url):
        self.base_url = base_url
        self.chat = self.Chat(self)

    class Chat:
        def __init__(self, client):
            self.completions = self.Completions(client)

        class Completions:
            def __init__(self, client):
                self.client = client

            def create(self, model, messages):
                payload = {
                    "messages": [
                        {"content": m["content"]} for m in messages
                    ]
                }

                r = requests.post(
                    self.client.base_url,
                    json=payload,
                    timeout=300
                )
                r.raise_for_status()
                data = r.json()

                # ---- OpenAI-like response object ----
                Message = type(
                    "Message",
                    (),
                    {"content": data["choices"][0]["message"]["content"]}
                )

                Choice = type(
                    "Choice",
                    (),
                    {"message": Message()}
                )

                Response = type(
                    "Response",
                    (),
                    {"choices": [Choice()]}
                )

                return Response()



client = CustomLLMClient(CUSTOM_API_URL)

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default='2025', help="coliee2025")
parser.add_argument("--dataset", type=str, default='test', help="train or test")
args = parser.parse_args()


RDIR = "./PromptCase/task1_"+args.dataset+"_"+args.data+"/processed"
WDIR = "./PromptCase/task1_"+args.dataset+"_"+args.data+"/summary_"+args.dataset+"_"+args.data+"_txt"

os.makedirs(WDIR,exist_ok=True)

files = os.listdir(RDIR)

for pfile in tqdm(files[:]):
    file_name = pfile.split('.')[0]
    if os.path.exists(os.path.join(WDIR, file_name+'.json')):
        # print(pfile, 'already exists')
        pass
    else:
        # print(pfile, 'does not exist')
        with open(os.path.join(RDIR, pfile), 'r') as f:
            long_text = f.read()
            f.close()
        if len(encoding.encode(long_text)) < 500:
            summary_total = long_text
        else:
            summary_total = ''
            length = int(len(encoding.encode(long_text))/3500) + 1
            # Loop through each line in the file
            for i in range(length):
                para = long_text[3500*i:3500*(i+1)]
                for x in range(1):
                    try:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=[
                                {"role": "user", "content": "Summarize in 50 words: " + para},
                            ]
                        )
                        summary_text = completion.choices[0].message.content
                        break
                    except Exception as e:
                        print("Error:", e)
                        summary_text = "" 
                summary_total += ' ' + summary_text
        
        with open(os.path.join(WDIR, file_name+'.txt'), 'w') as file:
            file.write(summary_total)
            file.close()

print('finish')