# Extract the following web URL text using BeautifulSoup and save the result in a file “input.txt”. Apply the following on the “input.txt” file
# https://en.wikipedia.org/wiki/Google

#  importing libraries
from bs4 import BeautifulSoup
import urllib.request
from itertools import islice

def search_spider():

    # extracting the given web URL text using BeautifulSoup
    url = "https://en.wikipedia.org/wiki/Google"             # web URL
    source_code = urllib.request.urlopen(url)
    soup = BeautifulSoup(source_code, "html.parser")
    body = soup.find('div', {'class': 'mw-parser-output'})
    file.write(str(body.text))

# storing the result in a file “input.txt”
file = open('input.txt', 'a+', encoding='utf-8')
search_spider()

file1 = open('input.txt')
file2 = open('in_150lines.txt', 'a')
lines = list(islice(file1, 150))
file2.writelines(lines)
file1.close()
file2.close()
print("\n Data saved to input.txt file successfully!!")










