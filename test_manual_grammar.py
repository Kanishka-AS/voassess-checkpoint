#!/usr/bin/env python3
"""
Manual grammar heuristic validation test with realistic speaking-assessment answers.

This test suite uses 40 realistic 150-250 word spoken responses covering:
- Clean natural/intermediate answers
- Clean beginner answers with natural speech features
- Subject-verb agreement errors
- Tense errors
- Article errors
- Be/auxiliary errors
- Preposition errors
- Mixed-error answers

Each answer is designed to be realistic for a speaking assessment:
- Natural fillers (um, uh, you know)
- Contractions
- Conjunctions and clauses
- Longer sentences with embedded errors
- Deliberately difficult correct constructions to test false positives

The goal is to establish a realistic precision baseline before expanding lexicons.
"""

import sys
import os
from collections import Counter

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grammar_heuristics import detect_learner_errors


# ── Category 1: Clean natural/intermediate answers (5) ─────────────────────
CLEAN_INTERMEDIATE = [
    """
    I've been studying English for about six years now, and I think I've made
    a lot of progress. When I first started, I couldn't speak at all, but now
    I can hold conversations on most topics. The hardest part for me is still
    pronunciation, especially words with 'th' sounds. My teacher says I need to
    practice more, so I've been watching YouTube videos and repeating phrases.
    I also try to read English news every morning, which helps with vocabulary.
    I'm planning to take the IELTS exam next year, so I'm practicing every day.
    My goal is to get a band 7, which is required for the university I want to
    attend. I know it's going to be challenging, but I'm determined to succeed.
    """,
    
    """
    Technology has completely changed the way we communicate with each other.
    Twenty years ago, people would write letters or make phone calls, but now
    we send messages instantly through apps like WhatsApp and Telegram. Social
    media has made it easier to stay in touch with friends and family who live
    far away. However, I think there are also some negative effects. People
    spend too much time looking at screens, and sometimes they forget to have
    real conversations. I try to limit my screen time to two hours a day, and
    I always put my phone away during meals. It's important to find a balance
    between using technology and being present in the moment.
    """,
    
    """
    The most memorable trip I ever took was to Japan last spring. I went with
    my brother, and we spent two weeks exploring Tokyo, Kyoto, and Osaka. The
    food was incredible, especially the sushi and ramen. I was surprised by
    how polite everyone was, and the public transportation was amazingly
    efficient. One of my favorite experiences was visiting a traditional
    tea ceremony, where we learned about the history and rituals. We also
    climbed Mount Fuji, which was exhausting but worth it for the view. I
    definitely want to go back someday, perhaps during the cherry blossom
    season. That trip taught me a lot about Japanese culture and made me
    appreciate how diverse the world really is.
    """,
    
    """
    I believe that education is the most powerful tool we have to change
    the world. When people are educated, they can think critically, solve
    problems, and make better decisions. Unfortunately, not everyone has
    access to quality education. In many countries, children still don't have
    schools to go to, and girls are often denied the opportunity to learn.
    I think we all have a responsibility to help change this. Even small
    contributions, like donating books or sponsoring a child's education,
    can make a big difference. I'm currently volunteering at a local tutoring
    program, where I help underprivileged students with their homework. It's
    one of the most rewarding things I've ever done.
    """,
    
    """
    Working from home has become much more common since the pandemic started.
    At first, I found it difficult to stay focused and motivated, but now I've
    developed a routine that works well for me. I wake up at the same time
    every day, get dressed as if I was going to the office, and set up my
    workspace in a quiet room. I also make sure to take regular breaks and
    go for a walk during lunch. The biggest advantage is the flexibility
    I have with my time. I can pick up my children from school and cook
    dinner without rushing. However, I do miss the social interaction with
    colleagues. Video calls just aren't the same as being in the same room.
    """,
]

# ── Category 2: Clean beginner answers with natural speech (5) ─────────────
CLEAN_BEGINNER = [
    """
    Um, I think I'm getting better at English, you know. When I started, I was
    really scared to speak. I would just, uh, stay quiet in class. But now I
    feel more confident. I practice every day with my friends. We speak English
    together during lunch. Sometimes I make mistakes, but that's okay. My teacher
    says mistakes are part of learning. So I just keep trying and I don't give up.
    I also watch English movies with subtitles. That helps me understand better.
    I like watching cartoons because they speak slowly. My favorite is Peppa Pig.
    It's easy to understand and it's funny too. I hope one day I can speak like
    a native speaker. That's my dream, you know?
    """,
    
    """
    I live in a small town near the river. It's very quiet and peaceful here.
    There are many trees and birds. I like walking in the morning. The air is
    fresh and clean. My house is not very big, but it's comfortable. I have
    a garden where I grow vegetables. I grow tomatoes, onions, and carrots.
    They taste better than the ones from the store. I also have a dog. His
    name is Bruno. He is very friendly and playful. He always follows me
    when I go for walks. I love living here because it's safe and everyone
    knows everyone. It's a good place to raise children.
    """,
    
    """
    My daily routine is very simple. I wake up at six in the morning every day.
    First, I brush my teeth and wash my face. Then I have breakfast. I usually
    eat bread with butter and drink a glass of milk. After breakfast, I go to
    work. I work in a small office near the market. I sit at my desk and answer
    phone calls. I like my job because it's not too difficult. I finish work
    at five in the evening. Then I go home and relax. I watch TV or read a
    book. I go to bed at ten. I like having a routine because it keeps me
    organized and helps me sleep better at night.
    """,
    
    """
    I think learning a new language is difficult, but it's also very interesting.
    I started learning English three years ago. At first, I didn't understand
    anything. The words were strange and the grammar was confusing. But my
    teacher was very patient with me. She explained everything slowly. I also
    used an app on my phone to practice. I practiced for thirty minutes every
    day. Slowly, I started to improve. Now I can understand most things. I can
    also speak a little. I still make mistakes, but I don't worry about it.
    I think the most important thing is to keep practicing and not be afraid
    to make mistakes. That's how we learn, right?
    """,
    
    """
    My best friend is someone I've known since childhood. We grew up together
    in the same neighborhood. We went to the same school and played together
    every day. Now we are adults and we both have jobs, but we still meet
    every weekend. We usually go to a cafe and talk for hours. We talk about
    work, relationships, and life in general. He is very kind and supportive.
    Whenever I have a problem, he listens and gives me good advice. I trust
    him completely. I think true friendship is rare and precious. I'm grateful
    to have him in my life. I hope our friendship lasts forever.
    """,
]

# ── Category 3: Subject-verb agreement errors (5) ──────────────────────────
SVA_ERRORS = [
    """
    My sister have a very interesting job. She work as a nurse in a big hospital.
    Every day she take care of many patients. She say that it is a challenging
    but rewarding profession. The hospital have many departments, and she rotate
    between them. Sometimes she work in the emergency room, which is very busy.
    My parents are very proud of her. They think she have chosen the right career.
    Her colleagues all respect her because she work very hard. She also teach
    new nurses sometimes. She really enjoy her work. I admire her dedication
    and her commitment to helping others. She is a true inspiration to me.
    """,
    
    """
    Everybody know that exercise is good for health. My friend go to the gym
    every morning. He say it helps him stay fit and energetic. His trainer
    give him a strict workout plan. The exercises include running, weightlifting,
    and stretching. He follow this routine every day without fail. His family
    also support him. They encourage him to continue. His parents think that
    regular exercise improve both physical and mental health. His sister
    sometimes join him for the running sessions. They believe that staying
    active is the key to a long and happy life. I think they are right.
    """,
    
    """
    The government have announced new policies to protect the environment.
    These policies aim to reduce pollution and promote renewable energy.
    Many people welcome these changes because they care about the planet.
    My neighbor, who is an environmental scientist, say that this is a good
    start. He think that more action is needed though. The companies in our
    area have already started implementing some of the measures. They install
    solar panels on their rooftops. They also encourage their employees to
    use public transport. Everyone have a role to play in protecting our
    environment, and I believe that small changes can make a big difference.
    """,
    
    """
    The team of researchers work on a groundbreaking project. They study the
    effects of climate change on marine life. Each member of the team bring
    their own expertise to the table. One of the scientists focus on coral
    reefs, while another study ocean currents. Together, they collect data
    and analyze the results. Their findings suggest that the ocean is warming
    faster than expected. The government have provided funding for their
    research. The public also show interest in their work. Many people follow
    their progress online. The researchers hope that their work will help
    protect the oceans for future generations.
    """,
    
    """
    My family have a tradition of cooking together on Sundays. My mother make
    delicious meals, and my father help with the preparation. My younger sister
    do the cleaning and setting the table. Everyone in the family participate
    in this activity. We enjoy spending time together in the kitchen. My parents
    say that cooking together strengthen our bond as a family. My grandmother
    sometimes join us. She share her old family recipes with us. The food always
    tastes amazing. This tradition have been passed down through generations.
    I hope to continue it with my own family one day. It is one of my favorite
    family activities.
    """,
]

# ── Category 4: Tense errors (5) ───────────────────────────────────────────
TENSE_ERRORS = [
    """
    Yesterday I go to the supermarket to buy some groceries. I need milk, eggs,
    and bread for breakfast. When I arrive at the store, I realize that I
    forget my shopping list at home. I try to remember what I need, but I
    can't recall everything. So I call my wife and ask her to tell me again.
    She say the list over the phone. I write it down on a piece of paper.
    Then I finish my shopping and go home. When I arrive home, I find that
    I buy the wrong type of milk. It was a frustrating experience. I should
    have checked the list more carefully before leaving.
    """,
    
    """
    Last summer, I visit my grandparents in the countryside. I stay with them
    for two weeks. Every morning, I wake up early and help my grandfather in
    the garden. We plant vegetables and water the flowers. My grandmother
    cook delicious meals for us. I also spend time with my cousins. We play
    games and go for long walks. One day, we decide to climb the hill behind
    their house. It is a beautiful experience. We watch the sunset from the
    top. I never forget that moment. It was one of the best summers of my
    life. I really enjoy my time there.
    """,
    
    """
    When I was a child, I want to become a pilot. I dream about flying planes
    and traveling to different countries. I watch documentaries about aviation
    and read books about famous pilots. I also build model airplanes. It was
    my favorite hobby. But as I grow older, I realize that my eyesight is not
    good enough for being a pilot. So I change my career plans. Now I work as
    an engineer, which is also interesting. But sometimes I still think about
    what it would be like to fly a plane. I never lose my love for aviation,
    even though I take a different path in life.
    """,
    
    """
    Last week, my boss tell me that I have to give a presentation. I feel very
    nervous because I never give a presentation before. I spend the whole
    weekend preparing my slides and practicing my speech. On the day of the
    presentation, I stand in front of the room and begin to speak. My hands
    shake, but I try to stay calm. To my surprise, the presentation go well.
    My colleagues ask many questions, and I answer them confidently. My boss
    congratulate me afterwards and say that I did a great job. I feel very
    proud of myself. I learn that I can do things I never thought I could.
    """,
    
    """
    Two years ago, I decide to learn how to play the guitar. I buy a guitar
    and start taking lessons. My teacher give me exercises to practice every
    day. At first, it is very difficult. My fingers hurt and I can't play
    chords properly. But I don't give up. I practice every day for at least
    an hour. Slowly, I improve. Now I can play many songs. I even perform
    at small gatherings with my friends. I never imagine that I would be
    able to play music. It is a wonderful feeling. I am glad I decided to
    learn this instrument.
    """,
]

# ── Category 5: Article errors (5) ─────────────────────────────────────────
ARTICLE_ERRORS = [
    """
    I want to buy new car, but I don't have enough money. My friend suggested
    that I should consider buying used car instead. He said that used car can
    be just as good as new car if you choose carefully. I looked at some cars
    online, but I couldn't decide which one to buy. I need car that is reliable
    and fuel efficient. My brother, who is mechanic, told me to check engine
    and tires before making decision. He also said I should take car for test
    drive. I think I will wait until next month when I get bonus. Then I will
    finally buy car. It's exciting to think about.
    """,
    
    """
    My mother is excellent cook. She can make amazing dishes from all over
    world. Every Sunday, she prepares large meal for entire family. She always
    uses fresh ingredients and secret spices. Her specialty is chicken curry,
    which is absolutely delicious. I tried to learn her recipe, but I can never
    get same taste. She says that cooking is art and you need practice to
    master it. When I have family over, I always ask my mother to cook. It
    is always memorable experience. Everyone enjoys her food and asks for
    second helping. I hope I can cook like her someday.
    """,
    
    """
    I have problem with my laptop. It keeps shutting down unexpectedly. I
    think there is issue with battery or maybe with operating system. I tried
    to fix it myself, but I couldn't find solution. So I took it to repair
    shop. The technician said that I need new battery and also need to update
    software. He said it would cost about hundred dollars. I decided to spend
    money because I need laptop for my work. Without laptop, I can't do my
    job properly. I hope the repairs will solve problem. I really don't want
    to buy new laptop because it's very expensive.
    """,
    
    """
    I saw interesting movie last night. It was about astronaut who travels to
    Mars. The astronaut faces many challenges during journey. He has to survive
    in harsh environment with limited resources. The movie shows how astronaut
    uses his intelligence and skills to overcome obstacles. I was really
    impressed by acting and special effects. The director did amazing job.
    My friend recommended this movie to me, and I'm glad I watched it. I think
    everyone should see this film because it gives important message about hope
    and determination. I would definitely watch it again.
    """,
    
    """
    I am student at local university. I study computer science and mathematics.
    My university has beautiful campus with old buildings and green gardens.
    I like walking around campus during breaks. The library is my favorite
    place on campus. It has huge collection of books and quiet study areas.
    I spend lot of time there studying and doing homework. My professors are
    very knowledgeable and helpful. They always encourage students to ask
    questions and participate in discussions. I have made many good friends
    at university. I feel lucky to be student here. It's great place to learn.
    """,
]

# ── Category 6: Be/auxiliary errors (5) ────────────────────────────────────
BE_AUX_ERRORS = [
    """
    I studying for my final exams right now. It very stressful because I have
    five exams in one week. My friends also studying hard. We usually studying
    together in the library. Yesterday, I studying for eight hours. I so tired
    that I fell asleep at my desk. My roommate come in and wake me up. He
    telling me that I should take a break and get some rest. He right. I can't
    studying effectively if I too exhausted. So now I taking regular breaks and
    drinking plenty of water. I feeling more energetic and focused. I hope I
    doing enough to pass all my exams. I want to get good grades.
    """,
    
    """
    My brother working on an important project at his office. He always busy
    these days. He coming home very late every night. I worried about his
    health because he not eating properly. My mother also worry. She telling
    him to take care of himself. He saying that he will finish the project
    next week. Then he going to take a vacation. I thinking that he needs
    a break. He working too hard. I hoping he listen to our advice. I don't
    want him to get sick. His health is more important than any project.
    """,
    
    """
    The children playing in the park right now. They happy and full of energy.
    Their mother watching them from the bench. She always making sure they
    are safe. The children running around and laughing. One of them climbing
    the slide while the other swinging on the swings. The weather beautiful
    today. The sun shining and there's a gentle breeze. I sitting nearby
    reading a book. I enjoying the peaceful atmosphere. It nice to see
    children having fun. They so carefree and innocent. I remembering when
    I was young and played in the park like that. Those were good times.
    """,
    
    """
    I planning to visit my grandmother next month. She living in a small village
    near the coast. I very excited to see her because I not seeing her for
    almost a year. She always happy when I visit. She cooking my favorite
    dishes and telling me stories about her childhood. I love listening to
    her stories. She very wise and has many interesting experiences. I learning
    a lot from her. She also teaching me how to cook traditional recipes. I
    wanting to learn everything she knows. I hoping she stay healthy and happy.
    I looking forward to my visit. It going to be wonderful.
    """,
    
    """
    The students preparing for their science fair. They working on various
    projects. One group building a volcano model. Another group creating a
    solar-powered car. The teacher helping them with their experiments. She
    explaining scientific concepts and answering questions. The students very
    enthusiastic about their projects. They spending many hours in the lab.
    The science fair going to be held next Friday. The students excited and
    nervous at the same time. They hoping their projects will win prizes.
    The parents also coming to see the fair. It going to be a great event.
    Everyone looking forward to it.
    """,
]

# ── Category 7: Preposition errors (5) ─────────────────────────────────────
PREPOSITION_ERRORS = [
    """
    I'm really interested on learning about artificial intelligence. My professor
    specializes on machine learning and neural networks. He is very capable to
    explain complex concepts in simple terms. I have been studying AI for about
    two years now. I'm good in programming, especially Python. My research
    focuses on natural language processing. I want to work on a company that
    develops AI solutions. My friend is also interested on this field, so we
    often discuss about our projects together. I believe AI will have a huge
    impact on society. I'm excited to be part on this technological revolution.
    """,
    
    """
    I arrived to the airport at 6 AM for my flight. I was afraid from missing
    my flight, so I came very early. I checked my bags and went to the gate.
    The flight was delayed because of bad weather. I waited for two hours.
    When I finally boarded the plane, I was relieved. The flight attendant
    was very friendly. She congratulated me for my upcoming wedding. I was
    surprised because I didn't tell anyone. She saw my ring and assumed I was
    getting married. I told her I was traveling to my hometown for a family
    reunion. She apologized and wished me a good trip. I thanked her and
    relaxed during the flight. It was a pleasant journey after all.
    """,
    
    """
    My friend is married with a doctor. They have two children. She is proud
    with her husband's achievements. He works at a big hospital and helps many
    patients. She is responsible of taking care of their children. She does
    a great job. Her children are similar with their father in many ways.
    They are both very intelligent and kind. She often talks about her family
    and how happy she is. I admire her dedication to her family. She is also
    different with most people I know. She is very calm and patient. Her
    husband depends of her a lot. They make a great team. I'm happy for them.
    """,
    
    """
    I have a problem with my computer. It's different with the one I had before.
    This new one is much faster, but it keeps crashing. I'm not capable to fix
    it myself. I need to take it to a technician. I'm angry for the situation
    because I just bought this computer last month. The warranty should cover
    the repairs. I called the customer service department. They were helpful,
    but I had to wait on hold for a long time. They said I should bring the
    computer to their service center. I'm going there tomorrow. I hope they
    can solve the problem quickly. I need my computer for work and studies.
    """,
    
    """
    I'm responsible of organizing the company's annual conference. It's a big
    event with hundreds of attendees. I'm good in planning events, so I'm
    confident about this task. I need to discuss about the schedule with the
    speakers. I'm also arranging the venue and catering. My boss is proud with
    my work. He said I'm doing an excellent job. I'm depending of my team to
    help me with the logistics. They are very capable to handle their tasks.
    We are different with other event organizers because we focus on quality
    and detail. I'm excited about the conference. I think it will be a success.
    """,
]

# ── Category 8: Mixed-error answers (5) ────────────────────────────────────
MIXED_ERRORS = [
    """
    My sister have always wanted to be a doctor. She study very hard every day.
    Yesterday she go to the hospital for her internship interview. She was very
    nervous, but she try to stay calm. The interview go well, and she think she
    did a good job. She is interesting on surgery, so she hope to specialize
    in that field. Her professors are proud with her. They say she have great
    potential. My parents also very happy for her. They always supporting her
    dreams. I think she will become excellent doctor one day. She has the
    dedication and compassion that every good doctor need. I'm so proud of
    her achievements. She is role model for me. I hope I can also succeed
    in my own career like she have.
    """,
    
    """
    Last summer, I go to beach with my family. We stay there for one week.
    The hotel was nice, and the staffs were friendly. Every morning, I wake
    up early and go for swim. The water is warm and clear. I see many colorful
    fishes. My brother also enjoy swimming. He is good in diving. My mother
    prefer to relax on the beach. She reading books and listening to music.
    My father like to explore the local area. He visit many interesting places.
    One day, we decide to go on a boat trip. It was amazing experience. We
    saw dolphins and sea turtles. I never forget that trip. It was one of
    the best holidays we ever have. I looking forward to going again next
    year. We already planning our next family vacation.
    """,
    
    """
    I have problem with my English pronunciation. I can't say some words
    correctly. My teacher help me with practice. She give me exercises to
    do at home. I practice every day for thirty minutes. Yesterday I practice
    saying words with 'r' and 'l' sounds. They are very difficult for me.
    I also have trouble with grammar. I make many mistakes when I speak.
    My teacher say I need to focus on subject-verb agreement. She explain
    the rules to me. I try to remember them, but I often forget. I also
    need to improve my vocabulary. I don't know enough words to express
    myself properly. I watch English movies and read books to learn more
    words. I hoping to improve quickly. I want to speak fluently like my
    teacher. She is best teacher I ever have.
    """,
    
    """
    My friend recommend me to watch new movie. He say it very interesting.
    So I go to cinema yesterday and watch it. The movie was about astronaut
    who travel to Mars. The special effects was amazing. I really enjoy the
    movie. The story was also good. It make me think about life and purpose.
    After the movie, I discuss about it with my friend. We had different
    opinions. He think the ending was perfect. I find it a bit disappointing.
    But we both agree that it was worth watching. I recommend this movie to
    everyone who like science fiction. It's one of the best movies I see
    this year. I planning to watch it again when it comes on streaming.
    The director did excellent job with this film.
    """,
    
    """
    I working on a important project for my company. It require a lot of
    research and analysis. My team work very hard to meet the deadline.
    We have many meetings to discuss about the progress. My manager is
    satisfied with our work. He said we doing a great job. I responsible
    for the data analysis part. It challenging because we have large dataset.
    I using different tools to analyze the data. My colleague help me with
    the technical aspects. She very knowledgeable. We make good progress
    every day. I confident that we will finish on time. The project have
    taught me many new skills. I grateful for this opportunity. I think
    it will be valuable experience for my career. I looking forward to
    completing it successfully.
    """,
]


def count_words(text):
    """Count words in a transcript."""
    return len(text.split())


def run_test():
    """Run the full validation test suite."""
    
    # Combine all test cases with labels
    test_cases = []
    
    for text in CLEAN_INTERMEDIATE:
        test_cases.append(("clean_intermediate", text.strip()))
    
    for text in CLEAN_BEGINNER:
        test_cases.append(("clean_beginner", text.strip()))
    
    for text in SVA_ERRORS:
        test_cases.append(("sva_errors", text.strip()))
    
    for text in TENSE_ERRORS:
        test_cases.append(("tense_errors", text.strip()))
    
    for text in ARTICLE_ERRORS:
        test_cases.append(("article_errors", text.strip()))
    
    for text in BE_AUX_ERRORS:
        test_cases.append(("be_aux_errors", text.strip()))
    
    for text in PREPOSITION_ERRORS:
        test_cases.append(("preposition_errors", text.strip()))
    
    for text in MIXED_ERRORS:
        test_cases.append(("mixed_errors", text.strip()))
    
    # Run tests
    results = []
    total_words = 0
    transcripts_with_issues = 0
    total_issues = 0
    issue_counter = Counter()
    
    print("=" * 80)
    print("GRAMMAR HEURISTIC — VALIDATION TEST SUITE")
    print("=" * 80)
    print()
    print(f"Total test cases: {len(test_cases)}")
    print("Categories: clean_intermediate(5), clean_beginner(5), sva_errors(5),")
    print("           tense_errors(5), article_errors(5), be_aux_errors(5),")
    print("           preposition_errors(5), mixed_errors(5)")
    print()
    print("=" * 80)
    print()
    
    for idx, (category, text) in enumerate(test_cases, 1):
        word_count = count_words(text)
        total_words += word_count
        
        issues = detect_learner_errors(text)
        
        if issues:
            transcripts_with_issues += 1
            total_issues += len(issues)
            for issue in issues:
                issue_counter[issue["rule_id"]] += 1
            
            print(f"TEST {idx:02d}: {category} ({word_count} words)")
            print(f"Heuristic issues detected: {len(issues)}")
            print()
            print("Detected issues:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue['rule_id']}")
                print(f"     Wrong:      {issue['wrong']}")
                print(f"     Correct:    {issue['correct']}")
                print(f"     Category:   {issue['category']}")
                print(f"     Confidence: {issue['confidence']}")
                # Show a snippet of the context
                context = issue.get('context', '')
                if context:
                    if len(context) > 80:
                        context = context[:77] + "..."
                    print(f"     Context:    \"{context}\"")
                print()
            print("=" * 80)
            print()
        else:
            print(f"TEST {idx:02d}: {category} ({word_count} words)")
            print("  ✅ No heuristic errors detected.")
            print()
            print("=" * 80)
            print()
    
    # Print summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print(f"Total transcripts:            {len(test_cases)}")
    print(f"Total words:                  {total_words}")
    print(f"Transcripts with issues:      {transcripts_with_issues}")
    print(f"Total heuristic issues found: {total_issues}")
    print()
    print("Issue count by rule ID:")
    for rule_id, count in sorted(issue_counter.items()):
        print(f"  {rule_id}: {count}")
    print()
    print("=" * 80)
    print("Test completed.")
    print()


if __name__ == "__main__":
    run_test()