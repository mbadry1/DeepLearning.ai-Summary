################################################
# File name: download.py                       #
# Author: Mahmoud Badry                        #
# Date created: 2/11/2018                      #
# Date last modified: 2/11/2018                #
# Python Version: 3                            #
# Purpose: Download all notes in PDF format    #
# Requirements: pypandoc >= 1.4                #
################################################
import pypandoc


def main():
    home_link = "https://raw.githubusercontent.com/mbadry1/DeepLearning.ai-Summary/master/"
    marks_down_links = {
        "Deeplearning.ai summary Homepage":
            home_link + "Readme.md",
        "01- Neural Networks and Deep Learning":
            home_link + "1-%20Neural%20Networks%20and%20Deep%20Learning/Readme.md",
        "02- Improving Deep Neural Networks Hyperparameter tuning, Regularization and Optimization":
            home_link + "2-%20Improving%20Deep%20Neural%20Networks/Readme.md",
        "03- Structuring Machine Learning Projects":
            home_link + "3-%20Structuring%20Machine%20Learning%20Projects/Readme.md",
        "04- Convolutional Neural Networks":
            home_link + "4-%20Convolutional%20Neural%20Networks/Readme.md",
        "05- Sequence Models":
            home_link + "5-%20Sequence%20Models/Readme.md",
    }

    # Extracting pandoc version
    print("pandoc_version:", pypandoc.get_pandoc_version())
    print("pandoc_path:", pypandoc.get_pandoc_path())
    print("\n")

    # Starting downloading and converting
    for key, value in marks_down_links.items():
        print("Converting", key)
        pypandoc.convert_file(
            value,
            'pdf',
            extra_args=['--latex-engine=xelatex', '-V', 'geometry:margin=1.5cm'],
            outputfile=(key + ".pdf")
        )
        print("Converting", key, "completed")


if __name__ == "__main__":
    main()
