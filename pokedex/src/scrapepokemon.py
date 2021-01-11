from bs4 import BeautifulSoup
import argparse
import requests
import os

parse = argparse.ArgumentParser()
parse.add_argument("-p", "--htmlPath", help="html pokemon file path")
parse.add_argument("-o", "--outPath", help="output path fro storing images")
args = vars(parse.parse_args())



soup = BeautifulSoup(open(args["htmlPath"]).read(),features="html.parser")
links = soup.find_all('div', class_ = "infocard")

print("[INFO] - Download started ......")
list_link = []
for link in links:
    url = link.a.span['data-src']
    name = os.path.basename(url)
    rc = requests.get(url)

    if rc.status_code != 200:
        print("Error downloading {}".format(name))
        continue

    f = open(args["outPath"] + name.lower(), "wb")
    f.write(rc.content)
    f.close()




