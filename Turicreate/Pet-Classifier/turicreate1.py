import turicreate as tc

# Load images (Note: you can ignore 'Not a JPEG file' errors)
data = tc.image_analysis.load_images('PetImages', with_path=True)

# From the path-name, create a label column
data['label'] = data['path'].apply(lambda path: 'dog' if '/Dog' in path else 'cat')

# Save the data for future use
data.save('cats-dogs.sframe')

# Explore interactively
data.explore()

