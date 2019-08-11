import pygame
import datetime
import random
from random import choice
import numpy as np
from PIL import Image
pygame.init()
chinese = ['浙','苏','沪','京','辽','鲁','闽','陕','渝','川']
number = ['0','1','2','3','4','5','6','7','8','9']
ALPHABET = ['A','B','C','K','P','S','T','X','Y']
number_lens = 5
def random_license_plate_text(chinese=chinese, ALPHABET=ALPHABET, number=number, number_lens=number_lens):
    car_text = []
    car_text.append(random.choice(chinese))
    car_text.append(random.choice(ALPHABET))

    for i in range(number_lens):
        temp = random.choice(number)
        car_text.append(temp)

    return car_text
def generate_car_license_plate():
    license_plate_text = random_license_plate_text()
    license_plate_text = ''.join(license_plate_text)
    # 保留前面的程序

    draw_plate_text = license_plate_text[0:2] + ' ' + license_plate_text[2:]
    # 保留前面的程序

    font_styles = ['notosansschinese']
    font_style = choice(font_styles)
    font_size = random.randint(50, 70)
    font = pygame.font.SysFont(font_style, font_size)
    font.set_bold(True)
    # 保留前面的程序

    ftext = font.render(draw_plate_text, False, (255, 255, 255))
    fontWidth = ftext.get_width()
    fontHeight = ftext.get_height()
    # 保留前面的程序
    img = pygame.image.load("background.jpg")
    img.blit(ftext, ((400-fontWidth) / 2, (100-fontHeight)/2))
    # 保留前面的程序

    pil_string_image = pygame.image.tostring(img,"RGB",False)
    line = Image.frombytes("RGB",(400,100),pil_string_image)
    license_plate_image = np.array(line)

    return license_plate_text, license_plate_image
if __name__ == '__main__':
    text, imageData = generate_car_license_plate()
    image = Image.fromarray(imageData)
    image.save(text + '.jpg')
    print(text, 'OK!')