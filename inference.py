import traceback

import openai
import os
import time
import threading
import json
import _thread
from tqdm import tqdm

from contextlib import contextmanager
from collections import defaultdict
import logging
logger = logging.getLogger(__name__)

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s -%(module)s:  %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S ')
logger.setLevel(logging.DEBUG)

use_azure = "HKUST_OPENAI_API_KEY" in os.environ
if use_azure:
    hkust_openai_api_key = os.environ["HKUST_OPENAI_API_KEY"]
    openai.api_base = "https://hkust.azure-api.net"
    openai.api_key = hkust_openai_api_key
    openai.api_type = "azure"
    openai.api_version = "2023-05-15"
else:
    try:
        openai.api_key = os.environ["OPENAI_API_KEY"]
    except KeyError:
        logger.error("Please set OPENAI_API_KEY environment variable")
        exit(1)

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):
    
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def add_prompt(item, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    query = item['question']
    prompt = prompt.replace('{query}', query)

    if item.get('output'): # background info
        backinfo = rmreturn(item['output'][0])
        prompt = prompt.replace('{background}', backinfo)

    return prompt


def augment_input(input, engine='gpt-4o-mini', task='statement', max_tokens=50, num_sequence=1, temp=0.8):
    outputs = []
    if use_azure:
        messages=[
            {"role": "system", "content": "You are an expert in paraphrasing texts."},
            {"role": "user", "content": "Restate the following " + task + " with different wordings: " + input['question']},
        ]
        logger.info(f"Calling augmented_input with input: {json.dumps(messages)}")
        completions = openai.ChatCompletion.create(
            engine=engine,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temp,
            n=num_sequence,
        )
        print(completions["choices"][0]["message"]["content"])
        for c in completions["choices"]:
            outputs.append(input.copy())
            outputs[-1]['question'] = c["message"]["content"]
        time.sleep(1)
    else:
        raise NotImplementedError
    return outputs


def clustering_prompt(items, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    cluster_prompts = []
    for item in items:
        query = item['question']
        backinfo = rmreturn(item['output'][0])
        item_prompt = prompt.replace('{query}', query)
        item_prompt += f' {backinfo}'
        cluster_prompts.append(item_prompt)

    cluster_prompts.append(prompt)
    return ' \n\n\n\n '.join(cluster_prompts)


def run_embeddings(input_text, engine='text-similarity-davinci-001'):
    logger.info("running embeddings")
    texts = [t.replace('\n', '') for t in input_text]
    outputs = openai.Embedding.create(input=texts, model=engine)['data']
    embeddings = [o['embedding'] for o in outputs]
    logger.info("embeddings done")

    return embeddings


def run_inference(inputs_with_prompts, engine, max_tokens, num_sequence=1, temp=0, prefix='', N_backgroud=1):
    logger.info("running inference")
    logger.debug(f"engine: {engine}")
    # logger.debug(f"inputs_with_prompts: {inputs_with_prompts}")

    completions = {"choices": []}
    outputs = []
    # N_backgroud = 2 # number of background info you want to set - Yuan Kangyu
    try:
        logger.info("Try to do inference")
        if use_azure:
            for inputs in tqdm(inputs_with_prompts, desc="Inference"):
                # prefix, statement = inputs.split("\n\n", 1)
                # messages = [
                #     {"role": "system", "content": prefix},
                #     {"role": "user", "content": statement},
                # ],
                outputs_for_one_input = ""
                for _ in range(N_backgroud):
                    inputs = inputs.replace("\n\n", " ")
                    messages = [
                        {"role": "system", "content": prefix},
                        {"role": "user", "content": inputs},
                    ],
                    # logger.info(f"Calling GPT with input: {json.dumps(messages)}")
                    completions = openai.ChatCompletion.create(
                        engine=engine,
                        max_tokens=max_tokens,
                        messages=[
                            {"role": "system", "content": prefix},
                            {"role": "user", "content": inputs},
                        ],
                        temperature=temp,
                        n=num_sequence,
                    )
                    # print("Response of GPT: " + completions["choices"][0]["message"]["content"])
                    # print("Everythign is fine...")
                    outputs_for_one_input += completions["choices"][0]["message"]["content"] + "\n\n"
                    time.sleep(1)
                outputs.append(outputs_for_one_input)
        else:
            completions = openai.Completion.create(
                engine=engine,
                max_tokens=max_tokens,
                prompt=inputs_with_prompts,
                temperature=temp,
                n=num_sequence, # num of returned sequence
                )
        # break
    except Exception as e:
        logger.debug(f"Error when calling run_inference: {traceback.format_exc()}")
        time.sleep(2)
    return outputs


def run_main(inlines, outfile, engine, prompt, max_tokens, prefix='', N_backgroud=1 ,n=1, temp=0, augment_input_conf={"engine": "none"}):

    if os.path.exists(outfile):
        outs = open(outfile, 'a', encoding='utf8')
        # skip the lines that have been processed, close for development - Yuan Kangyu
        num_lines = len(open(outfile, 'r').readlines())
        inlines = inlines[num_lines - 1: ]

    else: # not os.path.exists(outfile)
        outs = open(outfile, 'a', encoding='utf8')
        outs.write(json.dumps({"prompt": prompt}) + '\n')

    # pbar = tqdm(total = len(inlines))
    index = 0
    # pbar.update(index)
    augment_input_total = (0 if augment_input_conf['engine']=='none' else augment_input_conf['num_sequence'])+1
    while index < len(inlines):
        inputs, answers, augmented_inputs = [], [], []
        inputs_with_prompts = []
        for _ in range(20//augment_input_total):
            if index >= len(inlines): break
            augmented_inputs.append([inlines[index]])
            if augment_input_conf["engine"] != "none":
                augmented_inputs[-1].extend(augment_input(inlines[index], **augment_input_conf))
            for ai in augmented_inputs[-1]:
                input_with_prompt = add_prompt(ai, prompt)
                inputs_with_prompts.append(input_with_prompt)
            inputs.append(inlines[index]['question']) ## a string
            answers.append(inlines[index]['answer']) ## a list of strings
            # inputs_with_prompts.append(input_with_prompt)
            index += 1

        samples = defaultdict(list)
        outputs = run_inference(inputs_with_prompts, 
            engine, max_tokens, n, temp, prefix, N_backgroud)
        for j, output in enumerate(outputs):
            samples[j//(n*augment_input_total)].append(output)

        for i in range(len(inputs_with_prompts)//augment_input_total):
            outs.write(json.dumps({
                'question': inputs[i], 
                'answer': answers[i], 
                'output': samples[i],
                'augmented_inputs': [ai['question'] for ai in augmented_inputs[i]]}) 
                +'\n')

        # pbar.update(len(inputs_with_prompts))

    # pbar.close()
    outs.close()