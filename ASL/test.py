from pipeline import model_pipeline

test_image = r"ASL\assets\A\A1007.jpg"

result = model_pipeline.inference(test_image)
print(result)
