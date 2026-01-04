import re, json, requests
from urllib.parse import quote
from tqdm import tqdm
from run_refined import refined_ned
import time
import sys
import random
import argparse

def main():
    parser = argparse.ArgumentParser(description="Adapts a given Dataset to the format WikiSP expects. If the Dataset has weird field names, the code might have to be adjusted slightly.\n The general workflow:\nIf you have a single dataset to convert, run this script with the 'train' mode. Split the resulting converted dataset into train, dev, and val sets using 'split_dataset.py'. The train set will be ready after this. After this, run this script on the dev and test sets again using 'test' mode, to get the finished test.jsonl and dev.jsonl.")

    parser.add_argument(
            "--input_file_path",
            type=str,
            required=True,
            help="Path to the dataset to be converted."
        )
    parser.add_argument(
            "--jsonl_output_path",
            type=str,
            required=True,
            help="Path to output the jsonl result."
        )
    parser.add_argument(
            "--json_output_path",
            type=str,
            required=True,
            help="Path to output the json results"
        )
    parser.add_argument(
            "--adapt_mode",
            type=str,
            choices=["train", "test"],
            required=True,
            help="Decides what format the converted dataset will have. This must be one of 'test' and 'train'."
        )

    args = parser.parse_args()

    #path to input for training data (code might have to be adapted based on format)
    INPUT_PATH = args.input_file_path

    #path to output as jsonl file
    JSONL_PATH = args.jsonl_output_path

    #path to output as json file. this is converted from the jsonl file above, as wikisp expects json
    OUTPUT_PATH = args.json_output_path

    WIKIDATA_API = "https://www.wikidata.org/w/api.php"

    #train or test
    MODE = args.adapt_mode

    def execute_sparql(query, max_retries=5, base_delay=2):
        url = 'https://query.wikidata.org/sparql'
        delay = base_delay

        for attempt in range(max_retries):
            try:
                r = requests.get(
                    url,
                    params={'format': 'json', 'query': query},
                    timeout=30,
                    headers={"User-Agent": "Wikidata VA Analysis, Stanford OVAL"}
                )
                r.raise_for_status()

                data = r.json()
                if "boolean" in data:
                    res = data['boolean']
                else:
                    res = data["results"]["bindings"]

                # Save results
                try:
                    sparql_results.insert_one({"sparql": query, "results": res})
                except Exception:
                    pass

                return res

            except requests.exceptions.HTTPError as err:
                if r.status_code in (500, 400):
                    print(f"Caught {r.status_code} Server Error:", err)
                    return []
                elif r.status_code == 429:  # Too many requests
                    print(f"Rate limited, backing off for {delay}s...")
                    time.sleep(delay + random.uniform(0, 1))  # jitter
                    delay *= 2  # exponential backoff
                    continue
                else:
                    raise  # re-raise unexpected HTTP errors

            except (requests.exceptions.ReadTimeout,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.JSONDecodeError,
                    json.decoder.JSONDecodeError,
                    KeyError) as e:
                print(f"Non-fatal error: {e}, retrying in {delay}s...")
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2
                continue

        # If all retries failed
        return []

    def strip_prefixes(sparql):
        # find positions, -1 means "not found"
        j = sparql.find("SELECT")
        k = sparql.find("ASK")
        l = sparql.find("CONSTRUCT")

        # filter out -1
        positions = [pos for pos in (j, k, l) if pos != -1]
        if not positions:
            # no recognized query type, return as-is
            return sparql.strip()

        i = min(positions)
        return sparql[i:].strip()



    def find_ids(s):
        # Capture P and Q ids prefixed by wdt:, wd:, just P/Q if present
        pids = set(re.findall(r'wdt:P(\d+)', s))
        qids = set(re.findall(r'wd:Q(\d+)', s))
        # also search for full <http://www.wikidata.org/entity/P123> forms (rare)
        pids |= set(re.findall(r'entity\/(P\d+)', s))
        qids |= set(re.findall(r'entity\/(Q\d+)', s))
        # normalize to form like 'P19', 'Q12439'
        pids = {"P"+p for p in pids if not p.startswith("P")}
        pids = {p if p.startswith("P") else "P"+p for p in pids}
        qids = {"Q"+q for q in qids if not q.startswith("Q")}
        qids = {q if q.startswith("Q") else "Q"+q for q in qids}
        return list(pids), list(qids)


    def fetch_labels(ids):
        # ids is list like ["P19","Q123", ...] - wbgetentities accepts comma-separated
        if not ids:
            return {}
        ids_param = "|".join(ids)
        params = {
            "action":"wbgetentities",
            "format":"json",
            "ids": ids_param,
            "props":"labels",
            "languages":"en"
        }
        headers = {
            "User-Agent": "MyWikiProject/0.1 (luisdrayer@web.de)"
        }
        r = requests.get(WIKIDATA_API, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json().get("entities", {})
        labels = {}
        for k,v in data.items():
            lab = v.get("labels", {}).get("en", {}).get("value")
            if lab:
                # normalize to lowercase + underscores, remove problematic chars
                normalized = re.sub(r'[^0-9a-z_]', '', lab.lower().replace(" ", "_"))
                labels[k] = normalized
        return labels


    def replace_ids_with_labels(sparql, labels_map):
        # Replace wdt:Pxxx and wd:Qxxx occurrences with names if available.
        def repl_wdt(m):
            pid = "P"+m.group(1)
            name = labels_map.get(pid)
            return f"wdt:{name}" if name else f"wdt:{pid}"
        def repl_wd(m):
            qid = "Q"+m.group(1)
            name = labels_map.get(qid)
            return f"wd:{name}" if name else f"wd:{qid}"
        s = re.sub(r'wdt:P(\d+)', repl_wdt, sparql)
        s = re.sub(r'wd:Q(\d+)', repl_wd, s)
        return s


    def get_entity_label(qid, lang="en"):
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "format": "json",
            "props": "labels",
            "languages": lang
        }
        headers = {
            "User-Agent": "MyWikiProject/0.1 (luisdrayer@web.de)"
        }
        r = requests.get(WIKIDATA_API, params=params, headers=headers, timeout = 30)
        r.raise_for_status()
        data = r.json()
        entity = data.get("entities", {}).get(qid, {})
        labels = entity.get("labels", {})
        if lang in labels:
            return labels[lang]["value"]
        else:
            print("No english label found for qid: ", qid)
            return None


    def jsonl_to_json(jsonl_path, json_path):
        data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # skip empty lines
                    obj = json.loads(line)
                    data.append(obj)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)



    instruction_text = "Given a Wikidata query with resolved entities, generate the corresponding SPARQL. Use property names instead of PIDs."

    total_lines = 0
    print("Starting...")

    # Count total lines first
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(INPUT_PATH, 'r', encoding='utf-8') as fin, open(JSONL_PATH, 'w', encoding='utf-8') as fout:
        for line in tqdm(fin, total = total_lines, desc= "Processing Lines"):
            out = {}

            if not line.strip(): continue
            rec = json.loads(line)
            q = rec.get("question") or rec.get("input") or ""
            sparql_raw = rec.get("sparql","")
            #print("Raw Sparql: ", sparql_raw)
            sparql_no_prefix = strip_prefixes(sparql_raw)
            #print("No Prefix: ", sparql_no_prefix)

            # keep 'sparql' field as machine canonical (with PIDs/QIDs)
            # ensure it uses short wd:/wdt: notation (we already stripped PREFIX)
            canonical_sparql = sparql_no_prefix

            # find ids and fetch labels
            pids, qids = find_ids(canonical_sparql)
            ids_to_fetch = pids + qids
            labels = fetch_labels(ids_to_fetch) if ids_to_fetch else {}

            # build human-friendly output (replace P/Q -> labels when possible)
            human_sparql = replace_ids_with_labels(canonical_sparql, labels)
            
        
            #If question doesnt end with '?', add it.
            if q[-1] != '?':
                q += '?'

            #run refined on question, get corresponding label
            qid_set = refined_ned(q)
            qid_list = list(qid_set)
            label_list = []
            for qid in qid_list:
                label_list.append(get_entity_label(qid))

            input_field = f"Query: {q}\nEntities:"
            
            #Insert entities from refined into the query
            for i in range(len(qid_list)):
                input_field += f"{label_list[i]} with QID {qid_list[i]};"
                #Replace the entities in the question with Qids for given entities:
                human_sparql = human_sparql.replace(str(label_list[i]).lower().replace(" ", "_"), str(qid_list[i]))
            
            if MODE == "train":
                out = {
                    "id": rec.get("id", ""),
                    "input": input_field,
                    "output": human_sparql,
                    "instruction": instruction_text,
                    "sparql": canonical_sparql
                }
                
            else:
                #Get wikidata results:
                res = execute_sparql(canonical_sparql)
                question_only = input_field.replace("Query:", "", 1).split("\n")[0].strip().replace("Query:", "", 1).strip()

                out = {
                        "id" : rec.get("id", ""),
                        "utterance": question_only,
                        "sparql": canonical_sparql,
                        "results": res
                }

            #check if the document size would be larger than 16mb, as mongodb refuses such big docs
            MAX_DOC_SIZE = 16 * 1024 * 1024  # 16 MB

            json_bytes = json.dumps(out, ensure_ascii=False).encode("utf-8")
                
            if len(json_bytes) > MAX_DOC_SIZE:
                print(f"\n⚠️ Skipping oversized sample ({len(json_bytes)/1024/1024:.2f} MB): {rec.get('id', 'unknown')}")
                if "utterance" in out:
                    print("Utterance:", out["utterance"])
                elif "input" in out:
                    print("Input:", out["input"])
                else:
                    print("No utterance/input found.")
                print("-" * 80)
                continue

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    jsonl_to_json(JSONL_PATH, OUTPUT_PATH)

    print("done ->", OUTPUT_PATH)

if __name__ == "__main__":
    main()
