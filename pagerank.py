import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000
corpus = {"1.html": {"2.html", "3.html"}, "2.html": {"3.html"}, "3.html": {"2.html"}}

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    output = dict()
    if len(corpus[page]) == 0:
        for key in corpus:
            output[key] = 1/len(corpus)
        return output

    for key in corpus:
        output[key] = (1 - damping_factor)/len(corpus)

    prob = damping_factor/len(corpus[page])

    for pageLink in corpus[page]:
        output[pageLink] += prob

    return output




def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    firstSample = random.choice(list(corpus.keys()))
    nextSample = firstSample
    output = dict()
    for key in corpus:
        output[key] = 0

    for _ in range(1, n):
        output[nextSample] += 1
        dist = transition_model(corpus, nextSample, damping_factor)
        page = []
        weight = []
        for key in dist:
            page.append(key)
            weight.append(dist[key])
        nextSample = (random.choices(population=page, weights=weight))[0]

    for key in output:
        output[key] = output[key]/n
    return output


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    currentRank = dict()
    N = len(corpus)
    for key in corpus:
        currentRank[key] = 1/N
    run = True
    while run:
        diff = []
        newRank = dict()
        for key in currentRank:
            summation = 0
            for i in corpus:
                if key in corpus[i]:
                    summation += currentRank[i]/len(corpus[i])
                elif len(corpus[i]) == 0:
                    summation += currentRank[i]/N
            newRank[key] = ((1-damping_factor)/N) + (damping_factor * summation)
        
        for key in currentRank:
            diff.append(abs(currentRank[key] - newRank[key]))
        if all(i < 0.0015 for i in diff):
            run = False
        
        currentRank = newRank
    
    return currentRank
            


if __name__ == "__main__":
    main()