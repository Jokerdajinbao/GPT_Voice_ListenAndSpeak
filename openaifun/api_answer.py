import openai

#openai.api_base = 'https://key.langchain.com.cn/v1'
#openai.api_key = 'sk-kcfJcDXKztSEuMxaSqVjvuniMFIlz8HSr2xApuxivkNINiEc'

def api_answer(text):
    # send a ChatCompletion request to count to 100
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {
            'role': 'user',
            'content': text
            }
        ],
        temperature=0,
        stream=True  # again, we set stream=True
    )
    previous_full_reply = ""  # 用于存储上一次循环结束后的完整回复内容
    # create variables to collect the stream of chunks
    collected_chunks = []
    collected_messages = []
    # iterate through the stream of events
    print("\nGPT答复：")
    for chunk in response:
        collected_chunks.append(chunk)  # save the event response
        chunk_message = chunk['choices'][0]['delta']  # extract the message
        collected_messages.append(chunk_message)  # save the message
        full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
        added_characters = full_reply_content[len(previous_full_reply):]
        # 更新上一次循环结束后的完整回复内容
        previous_full_reply = full_reply_content
        #print(f"回答中: {full_reply_content}\n")
        print(f"{added_characters}", end="")
    print("\n")
    #full_reply_content = ''.join([m.get('content', '') for m in collected_messages])
    return full_reply_content