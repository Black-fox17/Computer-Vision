from pipeline import model_pipeline

test_image = r"ASL\test_images\A.jpg"

result = model_pipeline.inference(test_image)
print(result)
