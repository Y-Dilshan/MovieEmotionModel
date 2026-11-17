from predict import predict_emotions

test_movies = [

    """
    A young boy discovers magical powers
    and begins an exciting adventure.
    """,

    """
    A woman loses her family and struggles
    with sadness and loneliness.
    """,

    """
    A detective investigates a mysterious
    murder while facing dangerous enemies.
    """

]

for i, plot in enumerate(test_movies, 1):

    print(f"\nMovie Test {i}")

    predict_emotions(plot)