import re

def format_reward(response: str) -> float:
    #pattern = re.compile(r".*?<rewrite>.*?</rewrite>.*?<rewrite_2>.*?</rewrite_2>.*?", re.DOTALL)
    #format_match = re.fullmatch(pattern, response)
    #return 1.0 if format_match else 0.0

    matches_in_rewrite = re.findall(r"<rewrite>([^<]*)</rewrite>", response, re.DOTALL)
    matches_in_rewrite_t2 = re.findall(r"<rewrite_2>([^<]*)</rewrite_2>", response, re.DOTALL)

    score = 0.0

    if matches_in_rewrite != []:
        score+=0.5
    if matches_in_rewrite_t2 != []:
        score+=0.5

    return score


doc = """
output: <think>
Okay, let's tackle this query rewriting. The user is asking for the percentage gain recognized on commercial mortgages held for sale in 2009. First, I need to understand the core elements here. The key components are "commercial mortgages held for sale," "2009," and "percentage gain."

Looking at the methods provided, semantic rewriting might be useful here. The original question is pretty specific, so expanding on the entities could help. For example, "commercial mortgages held for sale" could be expanded to include related terms like "real estate finance," "mortgage-backed securities," or "financial instruments." Also, specifying the year 2009 might need context, like the economic climate or specific events that year affecting mortgage markets.

Another angle is query decomposition. The question is about a percentage gain, which might involve multiple steps: first finding the total gain, then calculating the percentage. But the user might be looking for a direct answer, so decomposition might not be necessary unless the documents are complex.

General rewriting could involve paraphrasing. Instead of "percentage gain recognized," maybe "profit margin on commercial mortgages" or "return on investment for held-for-sale mortgages." Also, considering intent refinement, the user might be interested in the financial performance of a specific company or industry in 2009, so adding that context could help.

Hypothetical document generation might be overkill here unless the goal is to create a sample document. The query understanding method could help by breaking down the terms, but the user already has a clear question.

I think semantic rewriting with expansion of entities and context would be best. Adding related terms like "real estate finance," "mortgage-backed securities," and mentioning the economic context of 2009 (like the housing market crisis) could help retrieve more accurate documents. Also, changing the phrasing to "What was the profit margin on commercial mortgages held for sale in the United States during 2009, considering the economic conditions of that year?" might better match the language used in financial reports.
</think>

</think>

<rewrite>What was the profit margin on commercial mortgages held for sale in the United States during 2009, considering the economic conditions of that year, including the housing market crisis and the impact on financial institutions' portfolios? Please provide details on how the gain was calculated, any relevant accounting standards applied, and the context of mortgage-backed securities during that period.</rewrite>
user
The following are the top documents (only show the first part of each document) retrieved by the rewritten query, ordered from high relevance to low relevance:
<doc1>we stopped originating commercial mortgage loans held for sale designated at fair value during the first quarter of 2008 and intend to continue pursuing opportunities to reduce these positions at appropriate prices . for commercial mortgages held for sale carried at the lower of cost or market , strong origination volumes partially offset sales to government agencies of $ 5.4 billion during 2009 . we recognized net gains of $ 107 million in 2009 on the valuation and sale of commercial<\doc1>
<doc2>during 2009 , securities transferred into level 3 from level 2 exceeded securities transferred out by $ 4.4 billion . total securities measured at fair value and classified in level 3 at december 31 , 2009 and december 31 , 2008 included securities available for sale and trading securities consisting primarily of non-agency residential mortgage-backed securities and asset- backed securities where management determined that the volume and level of activity for these assets had<\doc2>
<doc3>december 31 , 2009 , $ 397 million of the credit losses related to securities rated below investment grade . as of december 31 , 2009 , the noncredit portion of otti losses recorded in accumulated other comprehensive loss for non-agency residential mortgage-backed securities totaled $ 1.1 billion and the related securities had a fair value of $ 2.6 billion . the fair value of sub-investment grade investment securities for which<\doc3>
<doc4>average securities purchased under resale agreements increased to $ 4.69 billion for the year ended december 31 , 2011 from $ 2.96 billion for the year ended december 31 , 2010 . average trading account assets increased to $ 2.01 billion for the year ended december 31 , 2011 from $ 376 million for 2010 . averages benefited largely from an increase in<\doc4>
<doc5>interest-earning assets including unearned income in the accretion of fair value adjustments on discounts recognized on acquired or purchased loans is recognized based on the constant effective yield of the financial instrument . the timing and amount of revenue that we recognize in any period is dependent on estimates , judgments , assumptions , and interpretation of contractual terms . changes in these factors can have a significant impact on revenue recognized in any period due to changes in products , market conditions or industry norms . residential and commercial mortgage servicing rights we elect<\doc5>
<doc6>we stopped originating commercial mortgage loans held for sale designated at fair value in 2008 and continue pursuing opportunities to reduce these positions at appropriate prices . at december 31 , 2012 , the balance relating to these loans was $ 772 million , compared to $ 843 million at december 31 , 2011 . we sold $ 32 million in unpaid principal balances of these commercial mortgage loans held for sale carried at fair value<\doc6>
<doc7>commercial loans held for sale during 2013 and 2012 , the bancorp transferred $ 5 million and $ 16 million , respectively , of commercial loans from the portfolio to loans held for sale that upon transfer were measured at fair value using significant unobservable inputs . these loans had fair value adjustments in 2013 and 2012 totaling $ 4 million and $ 1 million , respectively , and were generally based on appraisals of<\doc7>
<doc8>residential mortgage-backed securities at december 31 , 2012 , our residential mortgage-backed securities portfolio was comprised of $ 31.4 billion fair value of us government agency-backed securities and $ 6.1 billion fair value of non-agency ( private issuer ) securities . the agency securities are generally collateralized by 1-4 family , conforming , fixed-rate residential mortgages . the non-agency securities are also generally collateralized by 1-4 family residential mortgages<\doc8>
<doc9>are allocated using appropriate statistical bases . total expense for repairs and maintenance incurred was $ 2.2 billion for 2011 , $ 2.0 billion for 2010 , and $ 1.9 billion for 2009 . assets held under capital leases are recorded at the lower of the net present value of the minimum lease payments or the fair value of the leased asset at the inception of the lease . amortization expense is computed using the straight-line method<\doc9>
<doc10>assets held under capital leases are recorded at the lower of the net present value of the minimum lease payments or the fair value of the leased asset at the inception of the lease . amortization expense is computed using the straight-line method over the shorter of the estimated useful lives of the assets or the period of the related lease . 12 . accounts payable and other current liabilities dec . 31 , dec . 31 , millions 2010 2009 .<\doc10>
Please further improve the rewritten query based on the retrieval result above. Generate a new rewritten query enclosing with separate <rewrite_2> and </rewrite_2>. Output can look like: <rewrite_2> the new rewritten query </rewrite_2>.
assistant
<think>
The initial rewritten query aimed to capture the context of commercial mortgages held for sale in 2009, including economic conditions and valuation methods. However, based on the retrieved documents, it's clear that the focus is more on the **net gains recognized in 2009** from the sale of these mortgages, rather than broader economic context or valuation standards. The top documents (doc1, doc2, doc6, and doc7) mention specific figures like "$107 million in 2009," "$5.4 billion in sales," and "fair value adjustments," which directly relate to the percentage gain.

To improve the query, we should **refine the focus** to explicitly ask for the **percentage gain** and include **key terms** that appear in the retrieved documents, such as "net gains," "valuation," "fair value," and "sales to government agencies." This would better align the query with the documents that contain the required data. Using **semantic rewriting** with intent refinement is appropriate here, as it helps direct the search toward specific financial figures and contexts mentioned in the documents.

The new query should:
- Directly ask for the **percentage gain**.
- Reference the **year 2009**.
- Include **key terms** like "net gains," "commercial mortgages held for sale," "fair value," and "sales to government agencies."
- Suggest the inclusion of **accounting standards** if relevant.

By incorporating these elements, the query becomes more precise and better aligned with the information in the retrieved documents.
</think>

<rewrite_2>What was the percentage gain recognized on commercial mortgages held for sale in 2009, specifically referencing the net gains of $107 million mentioned in the documents, and how did these gains relate to sales to government agencies and fair value adjustments? Please include details on the accounting standards or valuation methods used to calculate this percentage gain.</rewrite_2>
"""

print(format_reward(doc))

print(re.findall(r"<rewrite>([^<]*)</rewrite>", doc, re.DOTALL))
print(re.findall(r"<rewrite_2>([^<]*)</rewrite_2>", doc, re.DOTALL))