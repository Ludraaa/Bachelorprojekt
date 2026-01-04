import argparse
import requests
import json
from pymongo import MongoClient
from tqdm import tqdm
import datetime
import atexit
import subprocess
from utils import fill_template
from utils import safe_request_json
import time
import multiprocessing
import re
from urllib.parse import urlencode
#from location_silei import location_search --not found anywhere, ask on github
from mention_heuristics import location_search


#edit
import os
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch

def execute_sparql(query, sparql_results):
    #if sparql_results.find_one({"sparql": query}):
    #    return sparql_results.find_one({"sparql": query})["results"]
    
    url = 'https://query.wikidata.org/sparql'

    r = safe_request_json(url, params = {'format': 'json', 'query': query}, headers={"User-Agent":"Wikidata VA Analysis, Stanford OVAL"}) 

    if "boolean" in r:
        res = r['boolean']
    elif "results" in r:
        res = r["results"]["bindings"]
    else:
        print("No valid results found.")
        res = []

    try:
        sparql_results.insert_one({
            "sparql": query,
            "results": res
        })
    except Exception:
        pass

    return res



def get_name_from_qid(qid, qid_name_mapping):
    candidate = qid_name_mapping.find_one({"qid" : qid})
    # print(candidate)
    if candidate:
        return candidate["name"]
    
    else:
        url = 'https://query.wikidata.org/sparql'
        query = '''
    SELECT ?label
    WHERE {{
    {} rdfs:label ?label.
    FILTER(LANG(?label) = "en").
    }}
        '''.format(qid)
        print("processing QID {}".format(qid))
        
        r = safe_request_json(url, params = {'format': 'json', 'query': query}, headers={"User-Agent":"Wikidata VA Analysis, Stanford OVAL"})
        #print("QID Lookup:\n", r)
        

        #name = r["results"]["bindings"][0]["label"]["value"]
        #
        #print("Found {} with name {}".format(qid, name))
        #qid_name_mapping.insert_one({
        #        "qid": qid,
        #        "name": name
        #    }
        #)

        #return name


        # -------- robust extraction --------
        try:
            bindings = r.get("results", {}).get("bindings", [])
            if not bindings:
                return None

            name = bindings[0]["label"]["value"]
        except Exception:
            return None
        # ----------------------------------

        print(f"Found {qid} with name {name}")

        qid_name_mapping.insert_one({"qid": qid, "name": name})
        return name
        
SERVER_PORT = 6000
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def do_ned_for_dev(target_db, mode, qid_name_mapping):
    if mode == "refined":
        from refined.inference.processor import Refined
        #Changed from model_name="/data0/wikidata-workdir/models/refined"
        refined = Refined.from_pretrained(
                                    model_name="../Data/Models/refined",
                                    entity_set="wikidata",
                                    data_dir="../Data/Models/refined",
                                    download_files=True,
                                    use_precomputed_descriptions=True
                                    )

        def refined_ned(utterance):
            #print("NED Input:\n", utterance)
            spans = refined.process_text(utterance)
            #print("NED Immediate Output:\n", spans)
            output = set()
            for span in spans:
                if span.predicted_entity.wikidata_entity_id:
                    qid = span.predicted_entity.wikidata_entity_id
             #       print("QID:\n", qid)
                    wikidata_name = get_name_from_qid("wd:" + qid, qid_name_mapping)
              #      print("Wikidata name:\n", wikidata_name)
                    if wikidata_name is not None:
                        output.add((wikidata_name, qid))
            #print("NED Output:\n", output)
            return output    
        
        dev_set = list(target_db.find())
        for i in tqdm(dev_set):
            if "refined_ned_results" in i and i["refined_ned_results"]:
                continue  # skip already processed

            utterance = i["utterance"]
            pid_mapping_list = list(refined_ned(utterance))
            target_db.update_one({
                "_id": i["_id"]
            }, {
                "$set": {
                    "refined_ned_results": pid_mapping_list
                }
            })
    elif mode == "oracle":
        dev_set = list(target_db.find())
        pattern = r"wd:Q\d+"
        for i in tqdm(dev_set):
            utterance = i["utterance"]
            qid_list = re.findall(pattern, i["clean_sparql"])
            qid_list_tuples = [(get_name_from_qid(i), i.split(":")[1]) for i in qid_list]
            target_db.update_one({
                "_id": i["_id"]
            }, {
                "$set": {
                    "oracle_ned_results": qid_list_tuples
                }
            })
    else:
        raise ValueError


#reimplementation of evaluate_dev using batching, because it took 10 seconds per query
def evaluate_dev_batch(mode, model_path, target_db, oracle_or_refined, adapter_path=False, batch_size=1):
    dev_set = list(target_db.find())
    print("Loading Model..")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        local_files_only=True
    )

    from peft import PeftModel

    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Loaded model with adapter (LORA)")
        print(model.peft_config)
        print(model.active_adapter)
        # --- Verification: check if LoRA layers exist ---
        lora_layers = [n for n, _ in model.named_parameters() if "lora" in n.lower()]
        print(f"Found {len(lora_layers)} LoRA layers.")
        if len(lora_layers) == 0:
            print("⚠️ WARNING: No LoRA layers detected. Adapter may not have loaded correctly!")
        else:
            print("✅ LoRA adapter successfully attached.")


    model.to("cuda")

    print("Loaded successfully.")
    print("Loading Tokenizer..")

    tokenizer_path = model_path

    if model_path != "../Data/Models/llama7b_wwq_q7":
        tokenizer_path = "../Data/Models/Llama-2-7b-hf"

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True
    )
    print("Loaded successfully.")
    model.eval()
    
    #print("Model vocab size:", model.config.vocab_size)
    #print("Tokenizer vocab size:", len(tokenizer))

    if len(tokenizer) != model.config.vocab_size:
        print("⚠️ Mismatch detected! Resizing embeddings.")
        model.resize_token_embeddings(len(tokenizer))

    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id  # make sure the ID is set too
    

    tokenizer.padding_side = "left"
    stop_token_id = tokenizer.eos_token_id

    def build_prompt(utterance, pid_mapping_list):
        _input = fill_template('prompts/property-name-gen.input', {
            "query": utterance,
            "qid_list_tuples": pid_mapping_list
        })
        _instruction = fill_template('prompts/property-name-gen.instruction')

        return (
            "Below is an instruction that describes a task, paired with an input that provides further context.\n"
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{_instruction}\n\n### Input:\n{_input}\n\n### Response:"
        )

    # Filter out examples that already have predictions (optional)
    dev_set = [i for i in dev_set if "predictions" not in i or not any(p["model_path"] == model_path for p in i["predictions"])]
    print("Amount to be generated: ", len(dev_set))

    for i in tqdm(range(0, len(dev_set), batch_size), desc="Generating"):
        batch = dev_set[i:i + batch_size]
        prompts = []
    
        for item in batch:
            
            utterance = item["utterance"]
            pid_mapping_list = item["refined_ned_results"] if oracle_or_refined == "refined" else item["oracle_ned_results"]
            prompt = build_prompt(utterance, pid_mapping_list)
            prompts.append(prompt)
        
        #print(prompts[0])

        # Tokenize with explicit max_length to prevent silent truncation of SELECT/SPARQL
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=model.config.max_position_embeddings - 200,  # usually 1024 or 2048
            add_special_tokens=True
        )
        
        if False:
            # Assuming inputs is from tokenizer(...)
            first_input_ids = inputs["input_ids"][0]  # first prompt in the batch
            first_labels = inputs.get("labels", first_input_ids)  # default to input_ids if labels missing

            for i, (token_id, label_id) in enumerate(zip(first_input_ids, first_labels)):
                decoded = tokenizer.decode(token_id, skip_special_tokens=False)
                print(f"{i}th token:")
                print(f"Token: {token_id}")
                print(f"Decoded: {decoded}")
                print(f"Label: {label_id}\n")

        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        
        # Generate outputs
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"].to(model.device),
                attention_mask=inputs["attention_mask"].to(model.device),
                max_new_tokens=200,
                do_sample=False,
                top_p=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=stop_token_id,  # <- stop at EOS
            )
        

        if False:
            # Get the generated sequence (take first sample if batch size = 1)
            generated_ids = outputs[0].tolist()
        
            # Print each token ID and its decoded representation
            for i, token_id in enumerate(generated_ids):
                decoded = tokenizer.decode([token_id], skip_special_tokens=False)
                print(f"{i:03d} | Token ID: {token_id:<6} | Decoded: {repr(decoded)}")

        decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        #print("Decoded Outputs: ", decoded_outputs, flush=True)

        ''' 
        # Postprocess and store
        for j, output_text in enumerate(decoded_outputs):
            if '</s>' in output_text:
                output_text = output_text.split('</s>')[0]
            final_output = output_text.strip().split("### Response:")[-1].strip()

            item = batch[j]
            existing_predictions = item.get("predictions", [])
            prediction = {
                "mode": mode,
                "model_path": model_path,
                "sparql": final_output,
            }'''
        

        for j, output_text in enumerate(decoded_outputs):
            #print("iterating through decoded outputs")
            # Truncate if needed
            #print("Raw Output-------------------------------")
            #print(output_text, flush=True)
            #print("----------------------------------------")
            if '</s>' in output_text:
                output_text = output_text.split('</s>')[0]

            final_output = output_text.split('### Response:')[-1]
            final_output = final_output.strip()

            item = batch[j]
            existing_predictions = item.get("predictions", [])
            prediction = {
                "mode": mode,
                "model_path": model_path,
                "sparql": final_output,
            }


            target_db.update_one(
                {"_id": item["_id"]},
                {"$set": {"predictions": [prediction] + existing_predictions}}
            )

            print(item["utterance"], flush=True)
            print(final_output, flush=True)
            print()
    print("Evaluation Done")



def execute_predicted_sparql(sparql, name_to_pid_mapping, sparql_results, qid_name_mapping):
    # first, let's replace the properties
    
    # if ("wdt:instance_of/wdt:subclass_of" in sparql):
    #     print("HELPPPP\n\n\n\n\n")
    
    # print(sparql)
    sparql = sparql.replace("wdt:instance_of/wdt:subclass_of", "wdt:P31/wdt:P279")
    # print(sparql)
    
    
    url = 'https://query.wikidata.org/sparql'
    extracted_property_names =  [x[1] for x in re.findall(r'(wdt:|p:|ps:|pq:)([a-zA-Z_\(\)(\/_)]+)(?![1-9])', sparql)]
    #print("Extracted Property Names for: " + '\n' + sparql)
    #print(extracted_property_names)
    pid_replacements = {}
    for replaced_property_name in extracted_property_names:
        if not name_to_pid_mapping.find_one({"name" : replaced_property_name}):
            
            i = replaced_property_name.replace('_', ' ').lower()
            pid_query = """
                SELECT ?property ?propertyLabel WHERE {
                ?property rdf:type wikibase:Property .
                ?property rdfs:label "%s"@en .
                SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
            }"""% i
            
            #time.sleep(1)
            #response = requests.get(url, params={'format': 'json', 'query': pid_query})
            #response.raise_for_status()
            #data = response.json()
            #print("Request 1---------------------")
            data = safe_request_json(url, params={'format': 'json', 'query': pid_query}, headers={"User-Agent": "Wikidata VA Analysis, Stanford OVAL"})
            #print("PID Query:\n", pid_query)
            #print("Json from request:\n", data)
            #print("Request End-------------------")
            if 'results' in data and 'bindings' in data['results'] and len(data['results']['bindings']) > 0:
                # Extract the property ID from the response
                property_id = data['results']['bindings'][0]['property']['value']
                property_id = property_id.replace('http://www.wikidata.org/entity/', '')
                
                #print("found by sparql querying wikidata.. this is good")

                print("inserting {} for {}".format(replaced_property_name, property_id))
                name_to_pid_mapping.insert_one({
                    "name": replaced_property_name,
                    "pid": property_id
                })
            else:
                # try querying https://www.wikidata.org/w/api.php?action=wbsearchentities&search=songwriter&language=en&limit=20&format=json&type=property
                url = "https://www.wikidata.org/w/api.php"
                params = {
                    "action": "wbsearchentities",
                    "search": i,
                    "language": "en",
                    "limit": 20,
                    "format": "json",
                    "type": "property"
                }
                #encoded_url = url + "?" + urlencode(params)
                # print(encoded_url)
                #time.sleep(1)
                #response = requests.get(encoded_url)
                #data = response.json()
                #print("Request 2-----------------")
                data = safe_request_json(url, params, headers={"User-Agent": "Wikidata VA Analysis, Stanford OVAL"})
                #print("Query:\n", params)
                #print("Json from request:\n", data)
                #print("Request End---------------")
                
                if "search" in data and len(data["search"]) > 0:
                    property_id = data["search"][0]["id"]

                    #print("found by fallback 1 (api)")

                    print("inserting {} for {} by querying aliases for property".format(replaced_property_name, property_id))
                    name_to_pid_mapping.insert_one({
                        "name": replaced_property_name,
                        "pid": property_id
                    })
                else:
                    
                    print("CANNOT FIND PROPERTY: {} for SPARQL {}".format(replaced_property_name, sparql))
                    return [], sparql

        pid = name_to_pid_mapping.find_one({"name" : replaced_property_name})["pid"]
        pid_replacements[replaced_property_name] = pid
    
    def sub_fcn(match):
        prefix = match.group(1)
        value = match.group(2)
        
        return prefix + pid_replacements[value]
    
    sparql = re.sub(r'(wdt:|p:|ps:|pq:)([a-zA-Z_\(\)(\/_)]+)(?![1-9])', lambda match: sub_fcn(match), sparql)
        
    # next, we need to replace the domain entities
    extracted_entity_names =  [x[1] for x in re.findall(r'(wd:)([a-zA-PR-Z_0-9-]+)', sparql)]
    
    #print("Extracted Entity names for: " + sparql)

    #print(extracted_entity_names)
    qid_replacements = {}
    for extracted_entity_name in extracted_entity_names:
        if extracted_entity_name in ["anaheim_ca"]:
            qid_name_mapping.delete_many({
                "name": extracted_entity_name
            })
        
        found = False
        for i in qid_name_mapping.find():
            if i["name"] == extracted_entity_name and "qid" in i:
                
                #print("Found in QID mapping")

                found = True
                qid_replacements[extracted_entity_name] = i["qid"]
            elif i["name"].lower().replace(' ', '_').replace('/','_').replace('-', '_') == extracted_entity_name and "qid" in i:
                found = True

                #print("Found in QID mapping after normalizing")

                qid_replacements[extracted_entity_name] = i["qid"]
        
        
        
        if not found:
            #print("not found in QID mapping : do location search")
            try_location = location_search(extracted_entity_name.replace("_", " "))
            if try_location is not None:

                #print("Found by location search: " + try_location)

                try_location = "wd:" + try_location
                print("inserting {} for {}".format(try_location, extracted_entity_name))
                qid_name_mapping.insert_one({
                    "name": extracted_entity_name,
                    "qid": try_location
                })
                qid_replacements[extracted_entity_name] = try_location
            else:
            
            # trying querying https://www.wikidata.org/w/api.php?action=wbsearchentities&search=governor%20of%20oregon&language=en&limit=20&format=json
            # url = "https://www.wikidata.org/w/api.php"
            # params = {
            #     "action": "wbsearchentities",
            #     "search": extracted_entity_name.replace("_", " "),
            #     "language": "en",
            #     "limit": 20,
            #     "format": "json"
            # }
            # encoded_url = url + "?" + urlencode(params)
            # response = requests.get(encoded_url)
            # data = response.json()
            # time.sleep(1)
            
            # if "search" in data and len(data["search"]) > 0:
            #     found_entity_id = "wd:" + data["search"][0]["id"]
            #     qid_replacements[extracted_entity_name] = found_entity_id
            #     print("inserting {} for {}".format(found_entity_id, extracted_entity_name))
            #     qid_name_mapping.insert_one({
            #         "name": extracted_entity_name,
            #         "qid": found_entity_id
            #     })
            # else:
                
                print("CANNOT FIND ENTITY: {} for SPARQL {}".format(extracted_entity_name, sparql))
                return [], sparql
    
    def sub_entity_fcn(match):
        value = match.group(2)
        return qid_replacements[value]
    
    sparql = re.sub(r'(wd:)([a-zA-PR-Z_0-9-]+)', lambda match: sub_entity_fcn(match), sparql)
        
    # finally, we can execute
    prediction_results = execute_sparql(sparql, sparql_results)
    # time.sleep(1)
    return prediction_results, sparql
        
def compare_results(res1, res2):
    # each is a list of results
    if type(res1) is bool or type(res2) is bool:
        return res1 == res2

    
    res1 = [list(x.values()) for x in res1]
    res2 = [list(x.values()) for x in res2]
    if (res1 == res2):
        return True
    else:
        # print(res1, res2)
        return False

def safe_divide(x, y):
    if x == 0 and y == 0:
        return 0
    return x / y

def normalize(res):
    if isinstance(res, bool):
        return res

    out = []
    for item in res:
        # Skip empty dicts or None
        if not item or not isinstance(item, dict):
            continue
        if isinstance(item, dict):
            val = list(item.values())[0]["value"]
            out.append(val)
        else:
            out.append(str(item))
    return out


    

def execute_predictions(model_path, target_db, wwq, name_to_pid_mapping, sparql_results, qid_name_mapping, my_sparql_results, overwrite_existing=False, save_path=False, comparison_path=False):
    
    json_entries = [] #this is used for writing predictions to file

    def print_results(i, prediction, final_sparql):
        print(i["utterance"], flush=True)
        print(bcolors.WARNING + final_sparql + bcolors.ENDC, flush=True)
        #again, clean sparql
        #print(bcolors.OKBLUE + i["clean_sparql"] + bcolors.ENDC)
        if save_path:
            entry = {
                "dev_set_id": i["id"],
                "predicted_sparql": prediction,
                "executable_sparql": final_sparql,
                "results": i.get("results", [])
            } 
            #check if the document size would be larger than 16mb, as mongodb refuses such big docs
            MAX_DOC_SIZE = 16 * 1024 * 1024  # 16 MB

            json_bytes = json.dumps(entry, ensure_ascii=False).encode("utf-8")

            if len(json_bytes) > MAX_DOC_SIZE:
                print(f"\n  Model prediction result exceeded max length; saving without results.")
                entry["results"] = ["EXCEEDED_MAX_LENGTH"]
            json_entries.append(entry)

    exact_match = 0
    total = 0
    total_F1_score = 0
    local_wikisp_match = 0
    total_non_empty = 0
    

    for i in target_db.find():
        
        key = i["id"]
        #print(i)
        print(i["utterance"])

        if i["results"] == []:
             #print(i["utterance"])
             print("skipped")
             continue
            
        total += 1

        # see if we have an existing result:        
        found_prediction = None
        if not overwrite_existing and "prediction_results" in i:
            for existing_prediction in i["prediction_results"]:
                if model_path == existing_prediction["model_path"]:
                    found_prediction = existing_prediction
                    print("use existing results for {}".format(i["id"]))
                    break

        
        
        if found_prediction is not None and (not overwrite_existing or total < 800):
            #EDIT what is clean sparql
            #if found_prediction["final_sparql"] == i["clean_sparql"] or compare_results(found_prediction["results"], i["results"]):
            if compare_results(found_prediction["results"], i["results"]):
                print("Results identical!")
                exact_match += 1
            else:
                print("Results were not identical.")
                model_prediction = None
                for prediction in i["predictions"]:
                    if prediction["model_path"] == model_path:
                        model_prediction = prediction["sparql"]
                        break
                print("Prediction: ")
                print_results(i, found_prediction["final_sparql"], model_prediction)
            prediction_res = found_prediction["results"]
        else:
            found = False
            for prediction in i["predictions"]:
                if prediction["model_path"] == model_path:
                    
                    prediction_results, final_sparql = execute_predicted_sparql(prediction["sparql"], name_to_pid_mapping, sparql_results, qid_name_mapping)
                    #final_sparql = final_sparql.strip()

                    print("Getting Predictions:")    
                    print("Predicted:\n" + final_sparql, flush=True)
                    
                    import re
                    gold_sparql = re.sub(r'(?is)^.*?(select)', r'\1', i["sparql"]).strip()
                    print("Gold Answer:\n" + gold_sparql)
                    
                    to_be_compared = i["results"]
                    
                    if save_path:
                        entry = {
                            "dev_set_id": i["id"],
                            "predicted_sparql": prediction["sparql"],
                            "executable_sparql": final_sparql,
                            "results": prediction_results
                        }
                        
                        #check if the document size would be larger than 16mb, as mongodb refuses such big docs
                        MAX_DOC_SIZE = 16 * 1024 * 1024  # 16 MB
                        
                        import json
                        json_bytes = json.dumps(entry, ensure_ascii=False).encode("utf-8")

                        if len(json_bytes) > MAX_DOC_SIZE:
                            print(f"\n  Model prediction result exceeded max length; saving without results.")
                            entry["results"] = ["EXCEEDED_MAX_LENGTH"]

                        json_entries.append(entry)

                    #as the KB has changed, get results on their sparql first to compare  august 2025 results
                    if (wwq):
                        #print("Getting 2025 result of gold sparql..")
                        res_2025 = execute_sparql(i["sparql"], sparql_results)
                        
                        #if we evaluate on wwq, use 2025 results, as the saved results are old
                        to_be_compared = res_2025

                        #print("Predicted Sparql: ")
                        #print(final_sparql)
                        
                    if comparison_path:

                        #Get the prediction on the set, to see if the predictions are exactly the same
                        key = i["id"]
                              
                        local_prediction = my_sparql_results.find_one({"dev_set_id":key})["executable_sparql"]
                        
                        local_res = execute_sparql(local_prediction, sparql_results)

                        print("Comparing to prediction:\n" + local_prediction)
                        
                        if local_res == [] and prediction_results == []:
                            if final_sparql == local_prediction:
                                local_wikisp_match += 1
                                total_non_empty += 1
                            else:
                                print("Both Predictions empty with different query; skipping..")
                        
                        elif final_sparql == local_prediction or compare_results(local_res, prediction_results):
                            local_wikisp_match += 1
                            total_non_empty += 1
                        else:
                            total_non_empty += 1
                    
                    #print("Pred Res:\n", normalize(prediction_results))
                    #print("Gold Res:\n", normalize(to_be_compared))

                    if final_sparql == gold_sparql or compare_results(prediction_results, to_be_compared):    
                        print("Match", flush=True)
                        exact_match += 1
                    else:
                        print("Wrong", flush=True)

                    found = True
                    break

            if not found:
                print("{} no prediction".format(prediction))
                raise ValueError
                
            prediction_results_db = {
                "model_path": model_path,
                "final_sparql": final_sparql,
                "results": prediction_results
            }
            old_prediction_results = []
            if "prediction_results" in i:
                for old_prediction_result in i["prediction_results"]:
                    if old_prediction_result["model_path"] != model_path:
                        old_prediction_results.append(old_prediction_result)
            
            try:
                target_db.update_one({
                    "_id": i["_id"],
                }, {
                    "$set": {
                        "prediction_results": [prediction_results_db] + old_prediction_results
                    }
                })
            except Exception:
                pass
            
            prediction_res = prediction_results
        
        gold_res = to_be_compared

        #normalize, because qald10 has things other than urls and the var name differs
        pred = normalize(prediction_res)
        gold = normalize(gold_res)

        print("###########################PredRes###############################")
        print(pred)
        print("###########################GoldRes###############################")
        print(gold)
        print("#################################################################")
        


        if type(gold_res) == bool or type(prediction_res) == bool:
            total_F1_score += 1 if gold_res == prediction_res else 0
        else:
            
            #both gold and predicted results are emtpy -> 100% overlap between predictions!!
            if len(gold) == 0 and len(pred) == 0:
                total_F1_score += 1
                continue

            true_positive = [x for x in pred if x in gold]
            false_positive = [x for x in pred if x not in gold]
            false_negative = [x for x in gold if x not in pred]
            
            precision = safe_divide(len(true_positive), len(true_positive) + len(false_positive))
            recall    = safe_divide(len(true_positive), len(true_positive) + len(false_negative))
            if precision + recall == 0:
                this_f1 = 0
            else:
                this_f1 = 2 * precision * recall / (precision + recall)
            total_F1_score += this_f1

        print("accuracy: {}/{} = {}".format(exact_match, total, exact_match/total), flush=True)
        print("F1 = {}".format(total_F1_score / total), flush=True)
        if comparison_path: 
            print("local comparison acc: {}/{} = {}".format(local_wikisp_match, total_non_empty, local_wikisp_match/total_non_empty))
        print()

    #write our predictions to file
    if save_path:
        import json
        with open(save_path, "w", encoding="utf-8") as fd:
            json.dump(json_entries, fd, indent=2, ensure_ascii=False)
    if comparison_path:
        skipped_count = total - total_non_empty
        print(skipped_count, " comparison cases were skipped due to both results being empty!")

        return exact_match/total, total_F1_score/total, local_wikisp_match/total_non_empty
    else:
        return exact_match/total, total_F1_score/total, -1

def conversant_check(server_address):
        
    _input = "Monica S. Lam"
    _instruction = "Have you heard of this professor?"
    
    prompt = [
        "Below is an instruction that describes a task, paired with an input that provides further context.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Input:\n{}\n\n### Response:".format(_instruction, _input)
    ]
    
    
    output = requests.post(
        url="http://127.0.0.1:{}/completions".format(server_address),
        json={
            "engine": "llama",
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 500,
            "top_p": 1,
            "stop": ['\n', '</s>'],
        },
    )
    
    print(bcolors.WARNING + output.json()["choices"][0]["text"] + bcolors.ENDC)


import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Path to the folder containing model files")
    
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the folder containing dev.json and test.json of the desired dataset")

    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["dev", "test"],  # restrict to these two values
        default="dev",             # optional default
        help="Evaluation mode: 'dev' or 'test'",
        required=True
    )

    parser.add_argument("--get_current_results", type=bool, help="Set this to true to pull the current gold results. Otherwise, this will be evaluated against the dataset's (possibly outdated) results.", default=True)

    parser.add_argument("--save_path", default=False, help="The path at which the prediction results should be saved")

    parser.add_argument("--comparison_path", default=False, help="Use this to compare model predictions to another model's predictions (previously saved via setting the '--save_path' flag to a file.")
    
    parser.add_argument("--mongo_port", required=True, help="Port of the running mongodb session for this evaluation.")

    args = parser.parse_args()

    model_path = args.checkpoint_dir
    data_path = args.data_dir
    wwq = args.get_current_results
    save_path = args.save_path
    comparison_path = args.comparison_path
    mongo_port = args.mongo_port
    
    ##############Mongo Stuff#############
    client = MongoClient(f"mongodb://localhost:{mongo_port}/")
    webquestion_dev = client["wikidata-eval"]["dev"]
    webquestion_test = client["wikidata-eval"]["test"]
    qald_test = client["wikidata-eval"]["qald7_test"]
    qald_train = client["wikidata-eval"]["qald7_train"]
    name_to_pid_mapping = client["wikidata-eval"]["name_to_pid_mapping"]
    qid_name_mapping = client["wikidata"]["qid_naming_mapping"]
    sparql_results = client["sparql_results"]["sparql_results"]


    #to store original wikiSP predictions and local wikisp results for comparison
    my_sparql_results = client["mywikisp_sparql"]["original_sparql"]
    #####################################
    
    target_database = webquestion_dev if args.eval_mode == "dev" else webquestion_test

    #Check if mongodb is running
    try:
        client.admin.command("ping")
        print("MongoDB running.")
    except Exception as e:
        print("MongoDB not running: ", e)
    
    # Helper function to load JSONL
    def load_jsonl(file_path):
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # skip empty lines
                    data.append(json.loads(line))
        return data

    #insert data into mongodb
    if client["wikidata-eval"]["test"].find_one() == None:
        print("Inserting " + data_path + "test.json")
        client["wikidata-eval"]["test"].insert_many(json.load(open(data_path + "test.json")))
    else:
        print("########################################################")
        print("Reusing already inserted dataset. If you want to evaluate on a new dataset, kill the mongodb first (using Makefule)!!")
        print("########################################################")

    if client["wikidata-eval"]["dev"].find_one() == None:    
        print("Inserting " + data_path + "dev.json")
        client["wikidata-eval"]["dev"].insert_many(json.load(open(data_path + "dev.json")))

    if comparison_path:
        print("Inserting " + comparison_path)
        client["mywikisp_sparql"]["original_sparql"].insert_many(json.load(open(comparison_path)))

    #get pid mapping
    print("Doing NED:")
    do_ned_for_dev(target_database, "refined", qid_name_mapping)

    #Get predictions
    print("Getting predictions:")
    evaluate_dev_batch(model_path, model_path,  target_database, "refined", batch_size = 1)
    
    print("Executing Predictions to get stats:")
    #Compare Predictions to get statistics etc.
    ret1, ret2, ret3 = execute_predictions(model_path, target_database, wwq, name_to_pid_mapping, sparql_results, qid_name_mapping, my_sparql_results,  overwrite_existing=True, save_path=save_path, comparison_path=comparison_path) 

    print("#############################DONE#############################")
     
    result = {
        "acc": ret1,
        "f1": ret2,
        "local_acc": ret3
    }

    print(json.dumps(result))  # parent script can parse this


    return result

    # ==========================================================================================================
    # if you are evaluating a new model, then you should do thd following:
    
    # 1st: start the model server to listen to evaluation requests
    # start_server(model_path)
    # atexit.register(stop_server)
    
    # 2nd: get predictions for your target dataset with required NED data (mode is either "refined" or "oracle")
    # this will run through the target dataset set and record predictions from model
    # evaluate_dev(SERVER_PORT, model_path, model_path, qald_test, "refined")
    
    # 3rd, finally, get and execution results from the model and compare with existing results, compute statistics
    # execute_predictions(model_path, qald_test, overwrite_existing=True)

if __name__ == "__main__":
    main()
