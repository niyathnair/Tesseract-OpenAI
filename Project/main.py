import pytesseract
import PIL.Image
from pytesseract import Output
import cv2
import openai

myconfig = r"--psm 1 --oem 3"
text = pytesseract.image_to_string(PIL.Image.open("iitgand.jpg"), config= myconfig)
print(text)
# topic_search = input(text)
# topic_search = topic_search.replace(' ','+')

# print(topic_search)

# browser = webdriver.Chrome('chromedriver.exe')
# elements = "https://www.google.com/maps/place/"+topic_search
# print(elements)
# for i in range(1):
#     # elements = browser.get("https://www.google.com/maps/place/"+topic_search)
    # elements = "https://www.google.com/maps/place/"+topic_search
    # print(elements)

img =cv2.imread("iitgand.jpg")
height, width, _ = img.shape

boxes = pytesseract.image_to_boxes(img, config=myconfig)
# print(boxes)
for box in boxes.splitlines():
    box=box.split(" ")
    img=cv2.rectangle(img, (int(box[1]), height - int(box[2])), (int(box[3]), height - int(box[4])), (0, 220, 0), 2)

# data = pytesseract.image_to_data(img, config=myconfig, output_type=Output.DICT)
# amount_boxes = len(data['text'])
# for i in range (amount_boxes):
#     if float(data['conf'][i])> 85:
#         (x, y, width, height) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
#         img = cv2.rectangle(img, (x, y), (x+width , y+height), (0, 220, 0), 2)
#         img = cv2.putText(img , data['text'][i], (x, y+height+10), cv2.FONT_HERSHEY_PLAIN, 0.1, (0, 255, 0), cv2.LINE_AA)


# text = pytesseract.image_to_string(PIL.Image.open("logo.jpg"), config=myconfig)
# print(text)
# print(data['text'])

cv2.imshow("img", img)
cv2.waitKey(0)

# text = pytesseract.image_to_string(img, config=myconfig)
# print(text)
# Replace 'YOUR_API_KEY' with your actual API key
api_key = 'YOUR_API_KEY'

# Initialize the OpenAI API client
openai.api_key = api_key

# Example prompt
prompt = text

# Use the OpenAI API to generate text
response = openai.Completion.create(
    engine="text-davinci-002",  # You can choose a different engine if needed
    prompt=prompt,
    max_tokens=50,  # Adjust the length of the generated text
)

# Print the generated text
print(response.choices[0].text)