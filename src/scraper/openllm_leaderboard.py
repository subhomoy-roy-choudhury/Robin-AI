import argparse
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json


def get_json_format_data():
    url = "https://open-llm-leaderboard-open-llm-leaderboard.hf.space/"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    script_elements = soup.find_all("script")
    json_format_data = json.loads(str(script_elements[1])[31:-10])
    return json_format_data


def get_datas(data):
    for component_index in range(
        0, 50, 1
    ):  # component_index sometimes changes when they update the space, we can use this "for" loop to avoid changing component index manually
        try:
            result_list = []
            i = 0
            while True:
                try:
                    results = data["components"][component_index]["props"]["value"][
                        "data"
                    ][i]
                    columns = data["components"][component_index]["props"]["headers"]

                    model_fullname_index = (
                        columns.index("fullname")
                        if "fullname" in [col.lower() for col in columns]
                        else -3
                    )  # Will be used when we extract the model name from the data

                    try:
                        results_json = {
                            "T": results[0],
                            "Model": results[model_fullname_index],
                        }  # Set the first 2 manually because normally there is a link in the "Model" column

                        if (
                            len(columns) < 20
                        ):  # If there are less than 20 columns (this number can definetly change), we know that we are trying wrong component index, so breaking loop to try next component index.
                            break

                        for col_index, col_name in enumerate(columns[2:], start=2):
                            results_json[col_name] = results[col_index]

                    except (
                        IndexError
                    ):  # Wrong component index, so breaking loop to try next component index. (NOTE: More than one component index can give you some results but we must find the right component index to get all results we want.)
                        break
                    result_list.append(results_json)
                    i += 1
                except (
                    IndexError
                ):  # No rows to extract so return the list (We know it is the right component index because we didn't break out of loop on the other exception.)
                    return result_list
        except (KeyError, TypeError):
            continue

    return result_list

def main():
    parser = argparse.ArgumentParser(description="Scrape and export data from the Hugging Face leaderboard")
    parser.add_argument("-csv", action="store_true", help="Export data to CSV")
    parser.add_argument("-html", action="store_true", help="Export data to HTML")
    parser.add_argument("-json", action="store_true", help="Export data to JSON")
    
    args = parser.parse_args()

    data = get_json_format_data()
    finished_models = get_datas(data)
    df = pd.DataFrame(finished_models)

    if not args.csv and not args.html and not args.json:
        args.csv = True  # If no arguments are provided, default to CSV export

    if args.csv:
        df.to_csv("data/open-llm-leaderboard.csv", index=False)
        print("Data exported to CSV")

    if args.html:
        df.to_html("data/open-llm-leaderboard.html", index=False)
        print("Data exported to HTML")

    if args.json:
        df.to_json("data/open-llm-leaderboard.json", orient='records')
        print("Data exported to JSON")


if __name__ == "__main__":
    main()