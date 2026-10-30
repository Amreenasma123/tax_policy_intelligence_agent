from taxguard.schemas import UserRequest
from taxguard.safety.guards import validate_date, detect_injection
class RequestRouterAgent:
    def route(self, question, country_code, as_of_date, topic=None):
        flags=[]
        if not question or len(question.strip())<5: return None,["invalid_or_too_short_question"]
        if len(question)>2000: return None,["question_too_long"]
        if not validate_date(as_of_date): return None,["invalid_as_of_date"]
        flags.extend(detect_injection(question))
        try: req=UserRequest(question=question.strip(),country_code=country_code,as_of_date=as_of_date,topic=topic)
        except Exception: return None,flags+["invalid_request_schema"]
        return req,flags
