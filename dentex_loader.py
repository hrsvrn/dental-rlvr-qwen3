def get_dentex_dataset(split="train"):
    print(f"Loading Dentex dataset split: {split}...")
    try:
        # Most users for RLVR or detection use the 'quadrant_enumeration_disease' 
        # or similar configs. Check the HF page for the specific config name you need.
        ds = load_dataset(
            "ibrahimhamamci/DENTEX", 
            "quadrant_enumeration_disease", # <--- Specify config here
            split=split
        )
        
        print(f"Dataset columns: {ds.column_names}")
        ds = ds.filter(lambda x: x["image"] is not None)
        return ds
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None