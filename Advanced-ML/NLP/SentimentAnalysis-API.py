import paralleldots
paralleldots.set_api_key("1vos7bv1GFsREPqzP4k6d90zRZnPrngg7v4D1dBcRCk")
paralleldots.get_api_key()
text = """
A hackathon is a large scale design sprint-like event where students code to create or problem solve an idea from concept to prototype in under 36 hours. From inception, the goal of Nueva Hacks was to gather two hundred remarkable high school students together and empower them to become interested in computer science, biotechnology, robotics, and STEM.
I have been inspired by you and your team's revolutionary work at 23andMe to advance the field of genetic-testing and biotech. I have attached a document for the details of sponsorship levels and was wondering if 23andMe would like to sponsor the Nueva Hackathon? The sponsorship levels are on page 4 of the attached pdf.
Let me know if you have any questions? I really look forward to working with you and your team at 23andMe!
"""
"""text variable is a block of text I took from an email message I sent recently. 
I am trying to see what the likelyhood it classifys spam or not spam """

print(paralleldots.intent(text))

#Parallel dots API for sentiment analysis
#Adapted from https://www.paralleldots.com/
