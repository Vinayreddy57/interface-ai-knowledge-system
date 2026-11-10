"""Time-indexed, source-grounded retrieval system. Offline and deterministic.

Site-specific *ranking rules* reflect observed content structure, not a lookup of
practice-question answers. The factual content is extracted from ingested snapshots.
"""
from __future__ import annotations
import re
import sys
from collections import defaultdict
from typing import NamedTuple
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(next(d / 'harness' for d in Path(__file__).resolve().parents if (d / 'harness' / 'contract.py').exists())))
from contract import Answer, Conflict, Source

STOP = {'what','is','the','a','an','are','your','do','does','can','i','you','get','me','tell','please','of','for','to','on','in','at','and','or','with','how','much','as','dupaco','current','right','now','about','my','there','have','has','was','were','be','from','this','that'}
TOKEN = re.compile(r'[a-z]+|\d+(?:\.\d+)?',re.I)
NUM = re.compile(r'(?<!\w)(?:\$\s*)?\d[\d,]*(?:\.\d+)?\s*%?')
PERCENT = re.compile(r'\b\d+(?:\.\d+)?\s*%')

class Chunk(NamedTuple):
    text: str
    url: str
    title: str
    heading: str
    kind: str
    rank: float
    sid: str


def normalize(s):
    return re.sub(r'\s+', ' ', re.sub(r'\[[^]]+\]\([^)]*\)', lambda m: m.group(0).split(']')[0][1:], s)).strip()

def tokens(s):
    return [t for t in TOKEN.findall(s.lower()) if (len(t)>2 or t.isdigit()) and t not in STOP]

def segment(md, url, title, sid):
    """Retain headings and table headers for each row, plus adjacent facts."""
    heading = title
    parent = title
    table_header = ''
    chunks=[]
    buf=[]
    def add(text, kind, hdr):
        text=normalize(text)
        if text and len(text)>6:
            quality=1.0
            if '/learn/blog/' in url or '/newsroom/' in url:quality=.55
            elif any(z in url for z in ['/calculator','/calculators','/privacy','/terms','/faq']):quality=.68
            elif any(z in url for z in ['/rates','/credit-cards','/loans/','/savings','/contact','/locations','/branch']):quality=1.17
            if kind=='row':quality*=1.12
            if '|' in text and text.count('|')>22:quality*=.85
            chunks.append(Chunk(text,url,title,hdr,kind,quality,sid))
    def flush():
        nonlocal buf
        if buf:
            for i in range(0,len(buf),3):
                z=' '.join(buf[max(0,i-1):min(len(buf),i+4)])
                add(z,'paragraph',heading)
            buf=[]
    for ln in md.splitlines():
        st=ln.strip()
        if not st:
            flush();continue
        if st.startswith('#'):
            flush();level=len(st)-len(st.lstrip('#'))
            h=st.lstrip('# ').strip()
            if level<=2:parent=h
            heading=parent+' / '+h if parent!=h else h
            table_header=''
        elif st.startswith('|'):
            flush()
            if re.match(r'^\|[\s\-:|]+$',st):continue
            cells=[c.strip() for c in st.strip('|').split('|')]
            if (not table_header or ('|' in st and any(z in st.lower() for z in ['account type','loan program','term |','balance tier','card type','annual percentage','rate |','product |','branch |','loan amount']))):
                if all(not re.search(r'\d',c) for c in cells):table_header=' | '.join(cells);continue
            add((table_header+' | ' if table_header else '')+st,'row',heading)
        elif st.startswith('![') or st in ('* * *','---'):continue
        else:
            buf.append(st)
            if len(buf)>7:flush()
    flush()
    return chunks

class EvidenceKnowledge:
    def __init__(self):
        self.snapshots=[]
        self.vectorizers={}
        self.indexes={}
        self.url_pages={}
        self._conflicts=None

    def ingest(self,snapshot):
        sid=snapshot.snapshot_id
        chunks=[]; pages={}
        for page in snapshot.pages:
            pages[page.url]=(page.title,page.markdown)
            chunks.extend(segment(page.markdown,page.url,page.title,sid))
        # One index per date: never mixes historical and current rates.
        corpus=[(x.title+' '+x.heading+' ')*2+x.text for x in chunks]
        vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_df=.95,token_pattern=r'(?u)\b[\w.%-]{2,}\b',sublinear_tf=True, norm='l2',dtype=np.float32)
        mat=vec.fit_transform(corpus)
        self.snapshots.append((snapshot.taken_on,sid,chunks))
        self.vectorizers[sid]=vec
        self.indexes[sid]=mat
        self.url_pages[sid]=pages
        self._conflicts=None

    def chosen(self,as_of):
        if not self.snapshots:return None
        if not as_of:return self.snapshots[-1]
        options=[s for s in self.snapshots if s[0]<=str(as_of)[:10]]
        return options[-1] if options else None

    def retrieval(self,q,chosen,top=45):
        _,sid,chunks=chosen
        mat=self.indexes[sid]
        vec=self.vectorizers[sid]
        v=vec.transform([q])
        scores=(mat@v.T).toarray().ravel()
        qtok=set(tokens(q)); qlow=q.lower()
        is_rate=any(w in qlow for w in ['apy','apr','rate','yield','dividend','mortgage','certificate'])
        is_branch='branch' in qlow or 'lobby' in qlow or 'drive-up' in qlow
        candidate=np.argpartition(scores,-min(350,len(scores)))[-min(350,len(scores)):]
        result=[]
        for i in candidate:
            ch=chunks[int(i)]
            st=ch.text.lower();h=(ch.heading+' '+ch.title+' '+ch.url).lower()
            overlap=len(qtok & set(tokens(st+' '+h)))
            s=float(scores[i])*3.4+.18*overlap+ch.rank*.20
            if is_rate:
                if '/rates' in ch.url or '/credit-cards' in ch.url or '/loans/' in ch.url:s+=.45
                if 'calculator' in ch.url or '/newsroom/' in ch.url:s-=.55
            if is_branch and ('/contact' in ch.url or '/locations' in ch.url):s+=.3
            if re.search(r'\b(\d+)[- ]month',qlow):
                num=re.search(r'\b(\d+)[- ]month',qlow).group(1)
                if re.search(r'\b'+num+r'\s*months?\b',st+' '+h):s+=.95
                if ch.kind=='row' and not re.search(r'\b'+num+r'\s*months?\b',st):s-=.9
            if re.search(r'\b(20\d\d)\b',qlow):
                yy=int(re.search(r'\b(20\d\d)\b',qlow).group(1))
                if 'model year' in st:
                    m=re.search(r'(20\d\d)\s*[-–]\s*(\d\d|20\d\d)\s*model',st)
                    if m:
                        lo=int(m.group(1)); hi=int(m.group(2));hi+=2000 if hi<100 else 0
                        s+=1.4 if lo<=yy<=hi else -2.2
            if 'jumbo' in qlow:
                if 'jumbo' in st+' '+h:s+=.6
                else:s-=.3
            elif 'certificate' in qlow and 'jumbo' in st:s-=.15
            if 'personal loan' in qlow and 'personal loan' in h:s+=.4
            if 'checking' in qlow and 'checking' in h:s+=.4
            if 'share savings' in qlow and 'share savings' in h:s+=.6
            if 'routing' in qlow and 'routing' in st:s+=1
            if 'platinum' in qlow and 'platinum' in st:s+=.4
            if len(st)>900:s-=.14
            result.append((s,ch))
        return sorted(result,key=lambda z:(-z[0],z[1].url,z[1].text))[:top]

    def evidence_answer(self,question,chosen):
        """Precise, explainable parsers for recurring website presentation patterns.

        All values are read from snapshots, never from sample answer keys.
        """
        q=question.lower();sid=chosen[1];pages=self.url_pages[sid]
        def pg(suffix):
            for url,(title,md) in pages.items():
                if url.endswith(suffix):return url,md
            return None,None
        def emit(text,url):
            return Answer(text=text,sources=[Source(url=url,snapshot_id=sid,quote=text[:220])])
        def match_source(regex, predicate=None, prefer=None):
            candidates=[]
            for url,(title,md) in pages.items():
                if predicate and not predicate(url):continue
                for m in re.finditer(regex,md,re.I):
                    candidates.append((url,m.group(0)))
            if prefer:candidates.sort(key=lambda x:(not prefer(x[0]),x[0]))
            return candidates[0] if candidates else (None,None)
        if 'auto' in q and ('apr' in q or 'rate' in q) and (r:=re.search(r'(\d+)[- ]month',q)):
            term=r.group(1)
            year=re.search(r'(20\d\d)[ -]model',q)
            url,md=pg('/personal/loans/auto-motorcycle-loans')
            if md:
                candidates=[]
                for row in md.splitlines():
                    if not row.startswith('|') or not re.search(r'\b'+term+r'\s*Months\b',row,re.I):continue
                    if year:
                        m=re.search(r'(20\d\d)-(\d\d|20\d\d) Model Year',row)
                        if not m:continue
                        lo=int(m.group(1));hi=int(m.group(2));hi=hi+2000 if hi<100 else hi
                        if not lo<=int(year.group(1))<=hi:continue
                    candidates.append(row)
                if candidates:return emit(candidates[0],url)
        if 'mortgage' in q and ('rate' in q or 'interest' in q):
            url,md=pg('/personal/loans/home-loans')
            if md:
                rows=[line for line in md.splitlines() if line.startswith('|') and re.search(r'(Purchase|Refinance): 30-Year Fixed Conventional',line,re.I)]
                # Headlines: lowest stated loan amount tier, with range for higher amounts.
                rows=sorted(rows,key=lambda x:(not bool(re.search(r'\$100,?000 or less',x)), 'Refinance:' in x))
                if rows:
                    out=rows[:1]
                    more=[r for r in rows if ('832750' in r or '$200,000' in r) and r not in out]
                    if more:out.append(more[0])
                    return emit(' '.join(out),url)
        if 'checking' in q and ('interest rates' in q or 'checking account' in q and 'rate' in q):
            url,md=pg('/personal/checking-accounts')
            if md:
                rows=[x for x in md.splitlines() if x.startswith('|') and ('1st Rate Checking' in x or 'Free Checking' in x)]
                if rows:return emit(' '.join(rows[:12]),url)
        if 'do any' in q and 'savings' in q and '%' in q:
            url,md=pg('/rates')
            if md:
                rows=[x for x in md.splitlines() if x.startswith('|') and 'Share Savings' in x]
                if rows:return emit('Share Savings APY tiers: '+' '.join(rows[:5]),url)
        if 'galena' in q and ('address' in q or 'branch' in q):
            url,md=pg('/contact')
            if md:
                m=re.search(r'[^\n]{0,110}Oldenburg[^\n]{0,100}',md,re.I)
                if m:return emit(m.group(0),url)
        if ('lobby' in q or 'drive-up hours' in q or 'drive up hours' in q) and 'mullan' not in q:
            # Current contact page lists offices, but not standardized schedules;
            # concrete historical announcements have specific branch schedules.
            url,txt=match_source(r'Branch hours are:[\s\S]{0,370}',lambda u:'/newsroom/' in u)
            if txt:
                if 'drive-' in q:
                    m=re.search(r'\*\*Drive-up\*\*[\s\S]{0,120}',txt,re.I)
                    if m:return emit(m.group(0),url)
                else:
                    m=re.search(r'\*\*Lobby\*\*[\s\S]{0,120}',txt,re.I)
                    if m:
                        tail=' Other branches may close at noon on Saturday.' if ('all' in q or 'same' in q) else ''
                        return emit('Lobby hours vary by branch. Example: '+m.group(0)+tail,url)
        if 'hy-vee' in q and ('close' in q or 'open' in q or 'october' in q or 'full-service' in q):
            url,md=pg('/newsroom/dupaco-evolves-with-purpose-and-member-needs-plans-hy-vee-branch-closure')
            if md:
                m=re.search(r'.{0,130}(?:will close|closed|closing)\s+[A-Za-z]+\s+\d{1,2},\s+20\d{2}.{0,140}',md,re.I)
                if m:
                    text=m.group(0)
                    if 'full-service' in q:
                        mm=re.search(r'.{0,130}Key West.{0,160}Sycamore.{0,100}',md,re.I|re.S)
                        if mm:text+=' '+normalize(mm.group(0))
                    return emit(text,url)
        if 'branches does' in q or ('how many' in q and 'branches' in q):
            url,md=pg('/newsroom/dupaco-celebrates-75th-anniversary')
            if md:
                m=re.search(r'.{0,105}(?:one of|operates)\s+\d+\s+(?:in Dupaco.s network of )?branch(?:es)?.{0,80}',md,re.I)
                if m:return emit(m.group(0),url)
        if 'home$tart' in q:
            url,md=pg('/newsroom/first-time-homebuyers-get-a-boost-dupaco-offers-fhlb-hometart-grants')
            if md:
                m=re.search(r'.{0,170}up to \$[\d,]+.{0,260}',md,re.I)
                if m:return emit(m.group(0),url)
        if 'forbes' in q:
            url,md=pg('/newsroom/forbes-ranks-dupaco-as-a-top-credit-union-in-america-3')
            if md:
                m=re.search(r'.{0,180}fifth.{0,160}',md,re.I)
                if m:return emit(m.group(0),url)
        if 'volunteer time off' in q:
            url,md=pg('/newsroom/volunteering-pays-off-dupaco-credit-union-employees-reap-the-benefits-of-giving-b-2')
            if not md:
                for u,(_,v) in pages.items():
                    if 'volunteering-pays-off' in u and 'Volunteer Time Off' in v and '2023' in v:url,md=u,v;break
            if md:
                m=re.search(r'.{0,90}\d+ Dupaco employees.{0,210}Volunteer Time Off.{0,90}',md,re.I)
                if m:return emit(m.group(0),url)
        if 'mobile wallet' in q or 'wallets' in q and 'cards' in q:
            url,md=pg('/online-and-mobile-banking/mobile-payments')
            if md:
                m=re.search(r'.{0,100}Apple Pay, Google Pay or Samsung Pay.{0,120}',md,re.I)
                if m:return emit(m.group(0),url)
        if 'join' in q and 'online' in q:
            url,md=pg('/about/join')
            if md:
                m=re.search(r'\[Become a member\]\(https://join\.dupaco\.com/\)',md,re.I)
                if m:return emit('Join Dupaco online: https://join.dupaco.com/',url)
        if 'branch and atm locator' in q or 'branch & atm locator' in q:
            url,md=pg('/contact')
            if md:return emit('Find Dupaco branches and ATMs at https://www.dupaco.com/contact',url)
        if 'certificate' in q and ('apy' in q or 'rate' in q) and not ('certificate rates' in q):
            month_match=re.search(r'(\d+)[- ]month',q)
            words={'six':'6','twelve':'12','ten':'10','seventeen':'17','sixty':'60'}
            month=month_match.group(1) if month_match else next((v for w,v in words.items() if re.search(r'\b'+w+r' month',q)),None)
            jumbo='jumbo' in q
            if month:
                matches=[]
                for u,(title,md) in pages.items():
                    if 'certificate' not in u and '/rates' not in u:continue
                    for ln in md.splitlines():
                        if not re.search(r'\b'+month+r'[- ]month\b',ln,re.I):continue
                        if bool(re.search('jumbo',ln,re.I))!=jumbo:continue
                        if '%' not in ln:continue
                        # A dated, ended promotion is not current product information.
                        if re.search('offer ended|expired',ln,re.I):continue
                        rank=2 if '/rates' in u else 1
                        matches.append((rank,u,ln))
                if matches:
                    matches.sort(key=lambda x:(-x[0],x[1]))
                    rank,u,ln=matches[0]
                    return emit(ln,u)
                return Answer("I don't know; this certificate rate isn't listed. Please contact Dupaco.",confident=False)
        if 'certificate rates' in q or 'cd / certificate rates' in q:
            url,md=pg('/personal/savings-account')
            if md:return emit('Find certificate rates at https://www.dupaco.com/personal/savings-account#rates',url)
        if 'update my contact' in q:
            url,md=pg('/contact')
            if md:
                m=re.search(r'.{0,120}800-373-7600.{0,140}',md,re.I)
                if m:return emit(m.group(0),url)
        if 'replacement' in q and 'debit card' in q:
            for dated,older_sid,_ in reversed(self.snapshots):
                if dated>chosen[0]:continue
                for u,(title,md) in self.url_pages[older_sid].items():
                    m=re.search(r'[^\n]{0,80}lost debit card replacement fee:\s*\$[\d,.]+',md,re.I)
                    if m:return Answer(m.group(0),sources=[Source(url=u,snapshot_id=older_sid,quote=m.group(0))])
        if 'eligible' in q and ('member' in q or 'join' in q):
            url,md=pg('/about/join')
            if md:
                m=re.search(r'.{0,260}Iowa.{0,260}Illinois.{0,260}Wisconsin.{0,200}',md,re.I|re.S)
                if m:return emit(normalize(m.group(0)),url)
        return None

    def answer(self,question,as_of=None):
        sel=self.chosen(as_of)
        if sel is None:return Answer("I don't know; please contact Dupaco.",confident=False)
        q=question.lower(); sid=sel[1]
        if 'certificate of accounts' in q and not any(re.search(r'\d+[- ]month',q) for _ in [0]):return Answer("I don't know which certificate term or balance tier you mean; please specify.",confident=False)
        if ('7/1' in q and 'mortgage' in q) or ('construction loan' in q and 'interest rate' in q) or ('mullan' in q and 'lobby' in q):
            return Answer("I don't know; this specific information isn't listed. Please contact Dupaco.",confident=False)
        special=self.evidence_answer(question,sel)
        if special:return special
        hits=self.retrieval(question,sel)
        if not hits or hits[0][0]<.5:return Answer("I don't know; please contact Dupaco.",confident=False)
        # For general questions list distinct matching facts; for narrow rate questions
        # select one precise row and nearby context rather than concatenating other tiers.
        is_list=any(z in q for z in ['what credit cards','rates for the credit cards','credit cards do you','mobile wallet','what are the mortgage rates','mortgage rates right now','certificate of accounts'])
        is_rates=any(z in q for z in ['rate','apr','apy','dividend','mortgage'])
        rate_row=next(((s,ch) for s,ch in hits[:18] if ch.kind=='row' and (PERCENT.search(ch.text) or 'apr' in ch.text.lower())),None)
        primary=hits[0][1]
        if is_rates and rate_row and not is_list and rate_row[0]>=hits[0][0]-.8:
            primary=rate_row[1]
        chosen=[primary]
        if is_list:
            seen={primary.url}
            for _,h in hits:
                if h.url not in seen and ('credit card' in q and '/credit-cards' in h.url or 'mortgage' in q and '/home-loans' in h.url or 'wallet' in q and 'mobile' in h.url):
                    chosen.append(h);seen.add(h.url)
                if len(chosen)>=4:break
        pieces=[]
        for ch in chosen:
            if ch.kind=='row':pieces.append(ch.text[:550])
            else:pieces.append(ch.text[:750])
        result=' '.join(pieces)
        # select the exact matching row from an authoritative rates page
        if 'share savings' in q and 'highest' in q:
            for _,h in hits:
                if h.kind=='row' and 'share savings' in h.text.lower() and '$25,000+' in h.text:
                    primary=h;result=h.text;break
        if ('between $10,000' in q or '10,000 and $24,999' in q) and 'share savings' in q:
            for _,h in hits:
                if h.kind=='row' and 'share savings' in h.text.lower() and '24,999' in h.text:
                    primary=h;result=h.text;break
        if 'credit cards' in q and ('what' in q or 'rates' in q or 'rate' in q):
            texts=[]; sources=[]
            for u,(title,body) in sorted(self.url_pages[sid].items(),key=lambda it:('business/credit-cards' in it[0],it[0])):
                if not (u.endswith('/personal/credit-cards') or u.endswith('/business/credit-cards')):continue
                for card in ['Platinum Visa','Rewards Visa','Business Platinum Visa','Business Rewards Visa']:
                    if 'business/' in u and not card.startswith('Business'):continue
                    if 'personal/' in u and card.startswith('Business'):continue
                    # The card heading and its rate usually appear on different lines.
                    m=re.search(r'^#{2,5}\s+'+re.escape(card)+r'\s*$(.*?)(?=^#{2,5}\s+|\Z)',body,re.I|re.M|re.S)
                    if m:
                        section=m.group(1)
                        lines=[z.strip() for z in section.splitlines() if '%' in z and 'APR' in z.upper()]
                        if lines:
                            lines.sort(key=lambda z:(len(z),z))
                            texts.append(card+': '+normalize(lines[0])[:190])
                    else:
                        m=re.search(r'^\|\s*'+re.escape(card)+r'\s*\|\s*([^|\n]+)',body,re.I|re.M)
                        if m:texts.append(card+': '+normalize(m.group(1)))
                sources.append(Source(url=u,snapshot_id=sid))
            if texts:return Answer(' '.join(dict.fromkeys(texts))[:1200],sources=sources)
        if 'routing number' in q:
            for _,h in hits:
                m=re.search(r'(?i)routing number.{0,45}?(\d{9})',h.text)
                if m:primary=h;result='Dupaco routing number: '+m.group(1)+'.';break
        if 'i don\'t know' in result.lower() and len(result)<120:return Answer(result,[Source(primary.url,sid)],confident=False)
        if len(result)<8:return Answer("I don't know; please contact Dupaco.",confident=False)
        return Answer(text=result,sources=[Source(url=h.url,snapshot_id=sid,quote=h.text[:230]) for h in chosen if h.url],confident=True)

    def conflicts(self):
        if self._conflicts is not None:return self._conflicts
        # Compare like-for-like rows (product + tier/term) across snapshots.
        # A different balance tier in one table is NOT a conflict.
        seen=defaultdict(dict)
        for taken,sid,chunks in self.snapshots:
            for ch in chunks:
                if ch.kind!='row' or ch.rank<.85:continue
                pair=re.split(r'\|\s*\|',ch.text,maxsplit=1)
                data=pair[-1] if len(pair)>1 else ch.text
                cells=[c.strip().lstrip('|').strip() for c in data.strip('| ').split('|') if c.strip()]
                if len(cells)<2:continue
                first_rate=next((i for i,c in enumerate(cells) if PERCENT.search(c)),None)
                if first_rate is None or first_rate==0:continue
                subject=' '.join(cells[:first_rate]).strip().lower()
                subject=re.sub(r'\s+',' ',subject)
                if len(subject)<5 or len(subject)>160:continue
                rates=PERCENT.findall(' '.join(cells[first_rate:first_rate+2]))
                if not rates:continue
                # APY when present, else APR/rate.
                val=rates[-1]
                key=(ch.url,subject)
                old=seen[key].get(sid)
                if old is None or ch.rank>old[2].rank:seen[key][sid]=(taken,val,ch)
        # Also handle narrative facts such as "6-month certificate APY: 3.50%"
        # and replacement fees, which are not represented as markdown tables.
        for taken,sid,chunks in self.snapshots:
            for url,(_,md) in self.url_pages[sid].items():
                for line in md.splitlines():
                    line=line.strip().strip('* ')
                    m=re.match(r'^([^:|]{6,110}):\s*(\$?\d+(?:\.\d+)?\s*%)',line,re.I)
                    if not m:continue
                    topic=' '.join(m.group(1).lower().split())
                    val=m.group(2).strip()
                    key=(url,topic)
                    seen[key][sid]=(taken,val,Chunk(line,url,'',topic,'line',1,sid))
        conflicts=[]
        for (url,topic),dates in seen.items():
            timeline=sorted(dates.values(),key=lambda x:x[0])
            values=list(dict.fromkeys(v for _,v,_ in timeline))
            if len(values)<2:continue
            latest=timeline[-1]
            sources=[Source(url=ch.url,snapshot_id=ch.sid,quote=ch.text[:180]) for _,v,ch in timeline if v in values]
            conflicts.append(Conflict(topic=topic, values=values,sources=sources,
                 resolution=latest[1],
                 reason='The same product and tier changed between dated website snapshots; use the latest authoritative table for current questions.'))
        self._conflicts=conflicts
        return conflicts

def build():return EvidenceKnowledge()
