def parse_class_mapping_from_str(class_mapping: str) -> dict:
    class_mapping = class_mapping.strip()[1:-1]

    mapping = {}

    for pair in class_mapping.split(","):
        class_label, class_name = pair.split(":")
        mapping[int(class_label)] = class_name.strip().replace("'", "")

    return mapping
