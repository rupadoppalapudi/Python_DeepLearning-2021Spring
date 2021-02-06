# Web scraping

from bs4 import BeautifulSoup
import requests

html = requests.get("https://en.wikipedia.org/wiki/Deep_learning")
soupObj = BeautifulSoup(html.content, "html.parser")

# printing the title of the page
print("\nTitle of the page: ", soupObj.title.string)  # Printing the title of the page

# finding all the links in the page with ‘a’ tag
links = soupObj.find_all('a')

# saving all the links in the file
link_file = open("links_file.txt", "w")
for link in links:
    url = str(link.get("href"))
    if url != "None":
        link_file.write(url + "\n")

# printing the success message to user
print('\nLinks written to the file successfully!!')
link_file.close()
