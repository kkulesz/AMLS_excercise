import bz2

if __name__ == "__main__":
    file = "003918-3-0213_img.pkl.bz2"
    with bz2.open(file, "rb") as f:
        data = f.read()
    new_file_path = file[:-4]  # get rid of '.bz2' ending
    open(new_file_path, 'wb').write(data)
    print(f"Decompressed: {file}")
