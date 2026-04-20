import json
import sys


def transform_jsonl_to_json(input_path, output_path):
    data_out = []

    with open(input_path, "r", encoding="utf-8") as infile:
        for line_num, line in enumerate(infile, 1):
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[ERROR] Invalid JSON on line {line_num}: {e}")
                continue

            # Rename "question" -> "utterance"
            if "question" in obj:
                obj["utterance"] = obj.pop("question")

            # Remove "paraphrases"
            obj.pop("paraphrases", None)

            data_out.append(obj)

    # Write as formatted JSON array
    with open(output_path, "w", encoding="utf-8") as outfile:
        json.dump(data_out, outfile, ensure_ascii=False, indent=2)


def main():
    if len(sys.argv) != 3:
        print("Usage: python transform.py input.jsonl output.json")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    transform_jsonl_to_json(input_path, output_path)
    print(f"[OK] Wrote transformed JSON to: {output_path}")


if __name__ == "__main__":
    main()
