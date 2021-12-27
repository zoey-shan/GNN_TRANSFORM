from PIL import Image, ImageDraw

default_colors = ['#005fa7', '#884c96', '#d762a0', '#b6bb12', '#00adc4']


def draw_ribbon(data,
                output_path=None,
                data_height=30,
                data_width=500,
                padding=10,
                colors=default_colors):
    num_series = len(data)
    num_colors = len(colors)
    height = data_height * num_series + padding * (num_series + 1)
    width = data_width + padding * 2

    img = Image.new('RGB', (width, height))
    canvas = ImageDraw.Draw(img)
    canvas.rectangle((0, 0, width, height), fill='#ffffff')
    for i, series in enumerate(data):
        total = sum(series)
        x = padding
        y = data_height * i + padding * (i + 1)
        for j, d in enumerate(series):
            d_width = d / total * data_width
            canvas.rectangle((x, y, x + d_width, y + data_height),
                             fill=colors[j % num_colors])
            x += d_width
    if not output_path:
        img.show()
    else:
        img.save(output_path)