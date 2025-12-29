from sklearn.model_selection import train_test_split

def split_train_val_test(x, y, test_size=0.20, val_size=0.20, seed=42):
    x_train_temp, x_test, y_train_temp, y_test = train_test_split(
        x, y,
        test_size=test_size,
        random_state=seed,
        stratify=y
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train_temp, y_train_temp,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_temp
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == "__main__":
    from src.data.load_adult import load_adult

    x, y, meta = load_adult(return_meta=True)
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(x, y)

    print("Shapes:")
    print("  train:", x_train.shape)
    print("  val:  ", x_val.shape)
    print("  test: ", x_test.shape)

    print("\nPositive rates:")
    print("  train:", round(float(y_train.mean()), 4))
    print("  val:  ", round(float(y_val.mean()), 4))
    print("  test: ", round(float(y_test.mean()), 4))
