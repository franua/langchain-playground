import random
import questionary as q
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.ollama import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain


# from langchain import hub

# MODEL = "llama3:latest"
MODEL = "dolphin-mixtral:latest"
TEMPERATURE = 0.8
VERBOSE = True
PERSONS = [
    "a knowledgable and supportive AI",
    "Prof. Albus Percival Wulfric Brian Dumbledore, the headmaster of the wizarding school Hogwarts",
    "Eminem, the famous rapper",
    "Elizabeth II, the Queen of the United Kingdom and other Commonwealth realms",
    "Rick Sanchez, a misanthropic, alcoholic scientist from a famous cartoon series 'Rick and Morty'",
    "i wanna play, random please!",
]


def get_person():
    p: str = q.select("Who do you want to talk to?", choices=PERSONS).ask()
    return PERSONS[random.randint(0, len(PERSONS) - 2)] if p.find("random") != -1 else p


def get_question() -> str:
    return q.select(
        "What's your question?",
        ["Tell me an interesting fact about {subject}", "Explain {subject}"],
    ).ask()


def get_subject():
    return q.text("What's the subject of your question?").ask()


def create_ollama_instance(impersonate: str = "a knowledgable and supportive AI"):
    return Ollama(
        model=MODEL,
        system=f"You are a helpful assistant who speaks like {impersonate}",
        temperature=TEMPERATURE,
        verbose=VERBOSE,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )


def create_prompt_template(
    tempplate: str = "Tell me an interesting fact about {subject}",
) -> PromptTemplate:
    return PromptTemplate(
        input_variables=["subject"],
        template=tempplate,
    )


def create_llm_chain(ollama, prompt):
    return LLMChain(llm=ollama, prompt=prompt)


def main(person: str = None, question: str = None, subject: str = None):

    person = get_person() if person is None else person
    question = get_question() if question is None else question
    subject = get_subject() if subject is None else subject

    print()
    # impersonate = PERSONS[random.randint(1, 3)]
    # print(f"Hello, I am {impersonate}")
    print(f"Hello, I am {person}.\n")

    ollama = create_ollama_instance(person)
    prompt = create_prompt_template(tempplate=question)
    chain = create_llm_chain(ollama, prompt)

    # print("-" * 80 + "\n")
    # subject = input("What subject do you want to learn an interesting fact about?\n")
    # print()

    try:
        chain.invoke(subject)
        print("\n")
    except Exception as e:
        print(f"An error occurred: {e}")

    match q.select(
        "What do you want to do next?",
        ["change subject", "change question", "change person", "quit"],
    ).ask():
        case "change subject":
            main(person=person, question=question)
        case "change question":
            main(person=person)
        case "change person":
            main()

    # just some empty space for the output readability
    for i in range(3):
        print()


if __name__ == "__main__":
    main()
