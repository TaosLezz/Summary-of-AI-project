from together import Together
import base64
from PIL import Image
from io import BytesIO
client = Together(
    base_url='https://api.together.xyz/v1',
    api_key=api_key
)
response = client.images.generate(
    prompt="A 20-year-old Asian girl with long black hair, wearing a traditional silk dress, standing confidently beside a majestic tiger. The tiger has striking orange fur with bold black stripes and piercing amber eyes. They are in a serene forest with soft sunlight filtering through the trees, creating a mystical and harmonious atmosphere.",
    model="black-forest-labs/FLUX.1-schnell-Free",
    width=1024,
    height=768,
    steps=1,
    n=1,
    response_format="b64_json"
)
# print(response.data[0].b64_json)
b64_string = response.data[0].b64_json  

image_data = base64.b64decode(b64_string)

image = Image.open(BytesIO(image_data))

image.show()