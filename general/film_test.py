from general.make_film import imgs2File


def main():
    # imgs2File('../output/BreakoutNoFrameskip-v4/mfpp_sarfa/heat', '../output/BreakoutNoFrameskip-v4/mfpp_sarfa/film/heat')
    imgs2File('../output/MsPacmanNoFrameskip-v4/mfpp_sarfa/heat', '../output/MsPacmanNoFrameskip-v4/mfpp_sarfa/film/heat')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('system exit')
