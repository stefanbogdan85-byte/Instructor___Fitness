from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from evaluation.groq_llm import GroqDeepEval
from evaluation.report import save_report
import sys
from dotenv import load_dotenv
import httpx
import asyncio

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
THRESHOLD = 0.8

test_cases = [
    LLMTestCase(
        input="Arata-mi 3 exercitii pentru piept, nivel incepator, pe care le pot face acasa fara echipament."
    ),
    LLMTestCase(
        input="Care sunt cateva exercitii de stretching pentru zona lombara?"
    ),
    LLMTestCase(
        input="Creeaza un plan de antrenament de 3 zile pentru a imbunatati forta si rezistenta picioarelor."
    ),
]

groq_model = GroqDeepEval()

evaluator1 = GEval(
    name="RelevantaFitness",
    criteria="""
    Evaluează măsura în care răspunsul este relevant pentru cerința de fitness formulată de utilizator.

    Un răspuns relevant:
    1. Răspunde direct și explicit la cerința legată de fitness (exerciții, antrenament, mobilitate, recuperare, condiție fizică).
    2. Conține informații specifice domeniului fitness și activității fizice, evitând conținut general sau din alte domenii.
    3. Respectă intenția utilizatorului (ex: solicitare de antrenament, stretching, planificare sau explicații practice).
    4. Menține focusul pe rezolvarea cerinței, fără digresiuni sau informații irelevante.

    Scor între 0 și 1:
    0 = complet irelevant pentru cerința de fitness
    1 = complet relevant, clar și bine focalizat pe fitness
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator2 = GEval(
    name="BiasFitness",
    criteria="""
    Evaluează dacă răspunsul conține bias sau presupuneri nejustificate în contextul fitness.

    Analizează următoarele tipuri de bias:
    1. Bias de nivel:
       - Supraestimarea sau subestimarea capacității utilizatorului față de informațiile disponibile.
    2. Bias de siguranță:
       - Recomandări care ignoră riscurile, limitele fizice sau principiile de prevenire a accidentărilor.
    3. Bias de stil:
       - Limbaj excesiv de prescriptiv, autoritar sau motivațional, fără adaptare la contextul utilizatorului.
    4. Bias de generalizare:
       - Presupunerea că aceeași soluție este potrivită pentru toți utilizatorii, fără menționarea variațiilor individuale.

    Scor între 0 și 1:
    0 = bias semnificativ prezent
    1 = fără bias detectabil în contextul fitness
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)


async def _fetch_response(client: httpx.AsyncClient, message: str, max_retries: int = 2) -> dict:
    for attempt in range(max_retries + 1):
        response = await client.post(f"{BASE_URL}/chat/", json={"message": message})
        data = response.json()
        if data.get("detail") != "Raspunsul de chat a expirat":
            return data
        if attempt < max_retries:
            await asyncio.sleep(2)
    return data


async def _run_evaluation() -> tuple[list[dict], list[float], list[float]]:
    results: list[dict] = []
    scores1: list[float] = []
    scores2: list[float] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        for i, case in enumerate(test_cases, 1):
            candidate = await _fetch_response(client, case.input)
            case.actual_output = candidate

            evaluator1.measure(case)
            evaluator2.measure(case)

            print(f"[{i}/{len(test_cases)}] {case.input[:60]}...")
            print(f"  Relevanță: {evaluator1.score:.2f} | Bias: {evaluator2.score:.2f}")

            results.append({
                "input": case.input,
                "response": candidate.get("response", str(candidate)) if isinstance(candidate, dict) else str(candidate),
                "relevanta_score": evaluator1.score,
                "relevanta_reason": evaluator1.reason,
                "bias_score": evaluator2.score,
                "bias_reason": evaluator2.reason,
            })
            scores1.append(evaluator1.score)
            scores2.append(evaluator2.score)

    return results, scores1, scores2


def run_evaluation() -> None:
    results, scores1, scores2 = asyncio.run(_run_evaluation())
    output_file = save_report(results, scores1, scores2, THRESHOLD)
    print(f"\nRaport salvat in: {output_file}")


if __name__ == "__main__":
    run_evaluation()
