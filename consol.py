from recsys_utils import *

# Load movie list
movieList, movieList_df = load_Movie_List_pd()

def get_rating_choice():
    while True:
        choice = input("Enter your rating ([+] = Liked, [-] = Did not Like, [0] = Have not Seen, Enter = Save & Exit): ").strip()
        if choice == '+':
            return '1'
        elif choice == '-':
            return '0'
        elif choice == '0':
            return '0.5'
        elif choice == '':
            return None  # Signal to exit
        else:
            print("Invalid input. Try again.")

def main():
    # Prompt for starting movie index
    while True:
        try:
            n = int(input("Which movie would you like to begin rating first? "))
            if n < 0 or n >= len(movieList_df):
                print(f"Please enter a valid index between 0 and {len(movieList_df) - 1}")
            else:
                break
        except ValueError:
            print("Please enter a valid integer.")

    filename = input("Enter the file name to save ratings (e.g., ratings.txt): ")

    with open(filename, 'w') as file:
        file.write("import numpy as np\n")
        file.write("my_ratings = np.zeros(4778) + 0.5\n")
        current_n = n
        while True:
            try:
                # Get movie name using .loc
                movie_name = movieList_df.loc[current_n, "title"]
            except KeyError:
                print(f"No movie found at index {current_n}. Ending.")
                break

            print(f"\nRate movie {current_n}: {movie_name}")
            rating = get_rating_choice()

            if rating is None:
                break

            file.write(f"my_ratings[{current_n}] = {rating}\n")
            current_n += 1

    print(f"\nRatings saved to '{filename}'. Press Enter to exit.")
    input()

if __name__ == "__main__":
    main()


