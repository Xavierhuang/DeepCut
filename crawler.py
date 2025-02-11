from icrawler.builtin import GoogleImageCrawler
import os

# Keywords for each orientation
orientation_keywords = {
    'right': [
        "horse right side profile",
        "horse facing right side",
        "horse right side show"
    ],
    'left': [
        "horse left side profile",
        "horse facing left side",
        "horse left side show"
    ],
    'forward': [
        "horse front view",
        "horse facing camera",
        "horse head on view"
    ],
    'backward': [
        "horse rear view",
        "horse from behind",
        "horse back view"
    ]
}

# Create directories
os.makedirs("horse_orientation_dataset/right", exist_ok=True)
for orientation in ['left', 'forward', 'backward']:
    os.makedirs(f"horse_orientation_dataset/other/{orientation}", exist_ok=True)

# Filter parameters for consistent full-body shots
filters = dict(
    type='photo',
    size='large',
    color='color',
    license='commercial,modify',
    date=((2010, 1, 1), None)
)

for keyword in keywords:
    google_crawler = GoogleImageCrawler(
        storage={"root_dir": "horse_orientation_dataset/right"},
        feeder_threads=1,
        parser_threads=1,
        downloader_threads=4
    )
    
    google_crawler.crawl(
        keyword=keyword + " -group -herd -multiple -head -partial",  # Exclude partial views
        max_num=100,
        min_size=(300,300),
        max_size=None,
        file_idx_offset='auto',
        filters=filters
    )

print("\nDone downloading horse images!")
print("\nImportant: Image Selection Criteria:")
print("\nKEEP only images that show:")
print("- Full body of the horse (from head to tail)")
print("- Clear right side profile view")
print("- Single horse only")
print("- Professional quality, clear photos")
print("\nREMOVE images that show:")
print("- Partial views (just head, legs, etc.)")
print("- Multiple horses")
print("- Horses not in right profile")
print("- Blurry or poor quality photos")
print("- Drawings or artwork")
print("- Extreme angles or poses")
print("\nLocation: horse_orientation_dataset/right/")

