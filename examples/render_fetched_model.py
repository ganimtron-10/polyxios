from fury import actor, window

import polyxios


def main():
    """
    Example demonstrating how to use polyxios to fetch and read a remote model,
    then render it using the fury library.
    """
    model_name = "elepham.obj"
    print(f"Fetching '{model_name}'...")
    model_path = polyxios.fetch(model_name)
    print(f"Model successfully fetched to: {model_path}")

    print("Loading the model using polyxios...")
    poly = polyxios.read(model_path)

    vertices = poly.vertices
    faces = poly.connectivity.reshape(-1, 3)

    print("Creating surface actor...")
    mesh_actor = actor.surface(vertices, faces, colors=(0.8, 0.8, 0.8))

    print("Rendering the model...")
    window.show([mesh_actor])


if __name__ == "__main__":
    main()
