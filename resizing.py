from PIL import Image

def resize_image(input_image_path,
                 output_image_path,
                 size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    # print('The original image size is {wide} wide x {height} '
    #       'high'.format(wide=width, height=height))

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    # print('The resized image size is {wide} wide x {height} '
    #       'high'.format(wide=width, height=height))
    # resized_image.show()
    resized_image.save(output_image_path)
    print(output_image_path)
for i in range(501,526):
    resize_image(input_image_path="one_num/"+str(i)+'.jpg',
                 output_image_path="one_num/"+str(i)+'.jpg',
                 size=(40, 70))
