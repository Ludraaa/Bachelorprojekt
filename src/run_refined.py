from refined.inference.processor import Refined
import argparse
from pymongo import MongoClient
from tqdm import tqdm
import os
from utils import safe_request_json

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
 

def refined_ned(refined, utterance, qid_name_mapping):
    spans = refined.process_text(utterance)
    output = set()
    for span in spans:
        if span.predicted_entity.wikidata_entity_id:
            qid = span.predicted_entity.wikidata_entity_id
            if qid_name_mapping is not None:
                wikidata_name = get_name_from_qid("wd:" + qid, qid_name_mapping)
                if wikidata_name is not None:
                    output.add((wikidata_name, qid))
            else:
                output.add(qid)
    return output

def do_ned_for_dev(refined, target_db, mode, qid_name_mapping):
    if mode == "refined":

        dev_set = list(target_db.find())
        for i in tqdm(dev_set):
            if "refined_ned_results" in i and i["refined_ned_results"]:
                continue  # skip already processed

            utterance = i["utterance"]
            pid_mapping_list = list(refined_ned(refined, utterance, qid_name_mapping))
            target_db.update_one({
                "_id": i["_id"]
            }, {
                "$set": {
                    "refined_ned_results": pid_mapping_list
                }
            })

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--mongo_port", type=int, required=True)

    parser.add_argument("--mode", type=str, choices=["dev", "test"], required=True)
    
    args = parser.parse_args()

    mongo_port = args.mongo_port
    mode = args.mode

    refined_path = os.environ.get("REFINED_PATH", "/extern/data/Models/refined")
    refined = Refined.from_pretrained(
                                model_name=os.environ.get("REFINED_PATH", "/extern/data/Models/refined"),
                                entity_set="wikidata",
                                data_dir=os.environ.get("REFINED_PATH", "/extern/data/Models/refined"),
                                download_files=True,
                                use_precomputed_descriptions=True
                                )
    
    ##############Mongo Stuff#############
    client = MongoClient(f"mongodb://localhost:{mongo_port}/")
    webquestion_dev = client["wikidata-eval"]["dev"]
    webquestion_test = client["wikidata-eval"]["test"]
    qid_name_mapping = client["wikidata"]["qid_naming_mapping"]
    
    target_db = webquestion_dev if mode == "dev" else webquestion_test

    do_ned_for_dev(refined, target_db, "refined", qid_name_mapping)




if __name__ == "__main__":
    main()
