def get_query_generation_prompt(topic: str) -> str:

    return f"""
    **Your Role:** You are an AI assistant specialized in crafting search engine queries designed to find **clear, introductory, and explanatory content** on complex technical topics.

    **Your Objective:** Your goal is to generate diverse and effective search queries to locate web pages, articles, tutorials, and potentially simple PDF documents (like lecture notes or primers) that explain the fundamentals of **"{topic}"**. The content found should be suitable for someone learning the basics of the topic or seeking a general understanding. The ultimate aim is to gather materials for an LLM dataset focused on foundational knowledge, definitions, explanations, and examples, rather than highly specialized academic research.

    **Input topic:** The specific topic to focus on is: **"{topic}"**

    **Your Task & Required Output Format (CRITICAL):**
    You MUST generate a response that is **ONLY a single, valid JSON object**.
    This JSON object MUST have a single key: `"queries"`.
    The value associated with the `"queries"` key MUST be a JSON list (an array) of strings. Each string in this list should be a generated search query.

    *   **Strict JSON Output Example (if the topic were, for example, "Example Topic Name"):**
        ```json
        {{
          "queries": [
            "what is Example Topic Name basics",
            "introduction to Example Topic Name for beginners",
            "Example Topic Name tutorial (explanation OR overview)",
            "how does Example Topic Name work simple explanation"
          ]
        }}
        ```
    *   **ABSOLUTELY NO EXTRA TEXT:** Your response must start *exactly* with `{{` (an opening curly brace for the JSON object) and end *exactly* with `}}` (a closing curly brace for the JSON object).
    *   Do NOT include any conversational introductions (e.g., "Here is the JSON you requested:"), explanations, justifications, markdown formatting like ````json ... ```` tags, or any characters whatsoever outside of the JSON object itself.
    *   The JSON should be formatted with an indent of 2 spaces (this is for readability; the most important aspect is that it is valid JSON).

    **Core Strategies for Query Generation (You MUST Utilize These):**

    1.  **Format Flexibility (No Strict PDF Filter):**
        *   **Do NOT** automatically include `filetype:pdf` in most queries. Explanatory content is often found on standard web pages (HTML).
        *   Use `filetype:pdf` *only sparingly* and *specifically* when targeting document types that might contain introductory material in PDF format (e.g., adding `tutorial filetype:pdf`, `lecture notes filetype:pdf`, `primer filetype:pdf`).

    2.  **Topic Framing:**
        *   Use the provided topic "{topic}" accurately but frame it within an introductory context.

    3.  **Incorporating Explanatory Keywords:**
        *   Strategically include keywords that indicate introductory or explanatory content using `OR` (or `|`) and parentheses `()`.
        *   *Keywords to consider:* `introduction`, `explanation`, `explained`, `basics`, `fundamentals`, `overview`, `tutorial`, `definition`, `examples`, `"what is"`, `"what are"`, `"how does ... work"`, `"how do ... work"`, `guide`, `primer`, `"for beginners"`.
        *   *Example Structure:* `"{topic}" (introduction OR basics OR explained)`
        *   *Example Structure:* `"what is {topic}"` (Adjust grammar if needed based on the topic)
        *   *Example Structure:* `"what are [plural form of {topic} if applicable]"`

    4.  **Phrasing as Questions:**
        *   Generate some queries directly as questions that a learner might ask about "{topic}".
        *   *Example:* `"how does {topic} work?"`
        *   *Example:* `"what are the key concepts of {topic}?"`
        *   *Example:* `"examples of {topic}"`

    5.  **Targeting Appropriate Sources (Broader Scope):**
        *   Do **NOT** limit searches strictly to `.edu` or `.ac.*` domains, although they can still be useful for introductory course materials (`site:.edu`).
        *   Consider including queries targeting:
            *   Online encyclopedias: `site:wikipedia.org` (if topic is suitable)
            *   Reputable science/tech magazines/blogs (mentioning the *topic* specifically): `"{topic}" site:quantamagazine.org`, `"{topic}" site:technologyreview.com`
            *   Educational platforms or resources: `"{topic}" tutorial`, `"{topic}" coursera`
            *   Company educational resources/blogs: `"[Company Name relevant to {topic}]" "{topic}" basics`
        *   *Example Structure:* `"basics of {topic}" site:[relevant site e.g., quantamagazine.org]`
        *   *Example Structure:* `"introduction to {topic}" site:*.edu`

    6.  **Focusing on Specific Concepts (Introductory Level):**
        *   Generate queries for basic explanations of *key sub-concepts* related to "{topic}", framed simply. (You might need to infer some common sub-concepts if the topic is broad).
        *   *Example:* `"[Key Sub-Concept of {topic}]" explanation for beginners`
        *   *Example:* `"what is [Key Sub-Concept of {topic}]" simply explained`

    7.  **Avoiding Overly Academic Filters:**
        *   **Do NOT** use terms like `research`, `paper`, `study`, `thesis`, `dissertation`, `journal`, `conference paper`, or `intitle:"research paper"` unless specifically aiming for an introductory review article (which might use `review` or `overview`). The goal is explanation, not deep academic contribution.

    **Important Considerations You MUST Internalize:**

    *   **Prioritize Clarity and Simplicity:** The queries should aim to find content that explains "{topic}" clearly.
    *   **Target Audience for Content:** Assume the *reader* is a beginner or learner regarding "{topic}".
    *   **Content Format:** Be open to finding HTML pages, blog posts, tutorials, FAQs, and simple, explanatory PDFs.
    *   **Accuracy:** Seek factually accurate information.
    *   **Diversity:** Generate a *range* of query types as described above. Ensure you provide multiple queries.

    **Execution Mandate:**
    Based on the input topic "{topic}", immediately apply all instructions.
    Your entire output MUST be **ONLY** the specified JSON object, starting with `{{` and ending with `}}`.
    Ensure the JSON is well-formed (e.g., correct use of commas, quotes around keys and string values, proper nesting of list within object).
    The JSON object should have an indent of 2 spaces for readability, but strict adherence to the content structure is paramount. Do not output any text or markdown before the opening `{{` or after the closing `}}`.
    """


def get_query_evaluation_prompt(query_list_json_str: str, num_queries_to_select: int) -> str:
    return f"""
    **Your Role:** You are a highly specialized AI assistant functioning as an **Expert Search Query Evaluator and Selector**. Your expertise lies in analyzing lists of search queries and identifying a diverse, representative sample based on the underlying strategies used to construct them, specifically for finding **introductory and explanatory content**.

    **Your Objective:** Your goal is to analyze the provided list of search queries below (which were generated to find introductory/explanatory content) and select a **small, diverse, and representative subset of exactly {num_queries_to_select} queries**. This subset is intended for initial **manual testing** by the user to understand the effectiveness of different query strategies for finding foundational information.

    **The List of Queries to Analyze:**
    --- START OF QUERY LIST ---
        {query_list_json_str}
    --- END OF QUERY LIST ---
    (Note: The query list above is provided as a JSON string representing a list of query strings.)

    **Your Task & Required Output Format (CRITICAL):**
    1.  **Analyze:** Carefully examine the search queries provided in the list above. Identify the different strategies and techniques employed, focusing on those relevant to finding **introductory/explanatory content** (e.g., use of question phrasing, explanatory keywords, specific site targeting for general info, limited/targeted PDF use).
    2.  **Select:** Choose **exactly {num_queries_to_select} queries** from the provided list that, *as a group*, best represent the **diversity** of these introductory search strategies present in the full list.
    3.  **Output - Strict JSON Structure:**
        Your response MUST be **ONLY a single, valid JSON object**.
        This JSON object MUST have a single key: `"selected_queries"`.
        The value associated with the `"selected_queries"` key MUST be a JSON list (an array) containing **exactly {num_queries_to_select}** objects.
        Each object in this list MUST have the following two keys:
        *   `"query"`: (string) The full text of the selected query string from the input list.
        *   `"justification"`: (string) A brief explanation (1-2 sentences) explaining *why* this query was chosen and which specific *introductory search strategy* it represents (e.g., "Represents direct question format", "Chosen for using common explanatory keywords", "Selected for targeting a specific reputable non-academic site", "Represents targeted search for introductory PDF materials").

    *   **Strict JSON Output Example (if `{num_queries_to_select}` were 2 and input queries were sample queries):**
        ```json
        {{
          "selected_queries": [
            {{
              "query": "sample query string 1 from input list",
              "justification": "Example justification highlighting strategy for Query 1."
            }},
            {{
              "query": "sample query string 2 from input list",
              "justification": "Example justification highlighting strategy for Query 2."
            }}
          ]
        }}
        ```
        (Note: The number of objects in the `"selected_queries"` list must be exactly `{num_queries_to_select}`.)

    *   **ABSOLUTELY NO EXTRA TEXT:** Your response must start *exactly* with `{{` (an opening curly brace for the JSON object) and end *exactly* with `}}` (a closing curly brace for the JSON object).
    *   Do NOT include any conversational introductions (e.g., "Here is the JSON you requested:"), explanations of your general process, justifications, markdown formatting like ````json ... ```` tags, or any characters whatsoever outside of the JSON object itself.
    *   The JSON should be formatted with an indent of 2 spaces (this is for readability; the most important aspect is valid, structurally compliant JSON).

    **Key Criteria for Your Selection (Apply these to the provided list, focusing on introductory strategies):**
    *   **Strategy Diversity:** The final {num_queries_to_select} should showcase different techniques present in the list suitable for finding *explanatory* content.
    *   **Source Diversity (Implied by `site:`):** If possible, include queries targeting different types of domains relevant to introductory content.
    *   **Complexity Variation:** Include a mix of simple questions and queries with more keywords or site targets.
    *   **Representativeness:** The chosen {num_queries_to_select} should give the user a good *preview* of the *kinds* of **explanatory/introductory results** the different query types in the provided list might return.

    **Important Considerations:**
    *   You are NOT judging which query is objectively "best" overall for finding *any* content, but which ones best represent the *different ways to search for introductory content* in the given list.
    *   You are selecting a *sample set for testing purposes*.
    *   The most crucial factor is **diversity and representation** of the introductory strategies within your chosen {num_queries_to_select}.
    *   Stick strictly to selecting **exactly {num_queries_to_select}** queries.

    **Execution Mandate:**
    Immediately perform the analysis and selection based on the query list provided within "The List of Queries to Analyze" section above.
    Your entire output MUST be **ONLY** the specified JSON object, starting with `{{` and ending with `}}`.
    Ensure the JSON is well-formed (e.g., correct use of commas, quotes around keys and string values, proper nesting of list and objects).
    Do not output any text or markdown before the opening `{{` or after the closing `}}`.
    """
    
    

def get_search_results_filtering_prompt(original_query: str, search_results_json_str: str) -> str:
    return f"""
    **Your Role:** You are an AI assistant specialized in **Filtering Search Results for LLM Dataset Curation**, focusing on identifying suitable **introductory and explanatory content**.

    **Your Objective:** Evaluate **each individual search result** provided below to determine if it represents clear, accurate, informative, and relevant introductory content related to the context of **"{original_query}"**. Your goal is to select and return *only* the results that are suitable for building an LLM dataset focused on foundational knowledge.

    **Original Query (for context):**
        {original_query}

    **Search Results Sample to Filter:**
    (This is a JSON string containing a JSON list of objects, each typically with 'title', 'link'/'url', and 'snippet'/'description' keys.)
    --- START OF SEARCH RESULTS SAMPLE ---
        {search_results_json_str}
    --- END OF SEARCH RESULTS SAMPLE ---

    **Your Task & Required Output Format (CRITICAL):**
    1.  **Analyze Each Result:** Carefully examine **each search result** (title, URL/domain, description/snippet) provided in the sample list above.
    2.  **Apply Suitability Criteria:** For *each result*, determine if it meets **ALL** of the following criteria for inclusion:
        *   **Relevance & Focus:** The result must be clearly related to the theme of **"{original_query}"** or its directly related foundational concepts. The *primary focus*, inferred from the title and snippet, should be **explanatory or introductory**.
        *   **Content Nature:** It must seem to offer substantial **explanatory content** (like an article, guide, tutorial, comprehensive blog post, clear overview). It should *not* be just a link list, an abstract, a product page, purely metadata, or extremely brief content.
        *   **Source Suitability:** The source (inferred from domain/title) should be generally appropriate for providing **reliable and understandable introductory information**.
        *   **Usefulness for LLM:** The content appears genuinely useful for training an LLM to *understand and explain the basics* related to **"{original_query}"**.
    3.  **Output - Strict JSON List Structure:**
        Your response MUST be **ONLY a single, valid JSON list (an array)**.
        This list will contain objects, where each object represents a search result that **passed ALL the suitability criteria**.
        Each object in this output list MUST have the following three keys, using values from the original input search result:
        *   `"title"`: (string) The original title of the suitable search result.
        *   `"url"`: (string) The original URL of the suitable search result. If the input result object has a `"link"` key, use its value for this output key. If it has a `"url"` key, use its value. (Expect one of these to be present in the input object).
        *   `"description"`: (string) The original description or snippet of the suitable search result. If the input result object has a `"snippet"` key, use its value for this output key. If it has a `"description"` key, use its value. (Expect one of these to be present in the input object).

    *   **If NO results from the input sample pass ALL the criteria, your output MUST be an empty JSON list: `[]`**

    *   **Strict JSON Output Examples:**
        *   If suitable results are found:
            ```json
            [
              {{
                "title": "Intro to Topic X",
                "url": "https://example.com/intro-x",
                "description": "A great starting point for understanding Topic X and its fundamentals."
              }},
              {{
                "title": "Topic X Explained Simply",
                "url": "https://another-site.org/topic-x-basics",
                "description": "This article breaks down Topic X for beginners."
              }}
            ]
            ```
        *   If NO suitable results are found:
            ```json
            []
            ```

    *   **ABSOLUTELY NO EXTRA TEXT:** Your response must start *exactly* with `[` (opening square bracket for the JSON list) if results are present, or `[]` if no results are present, and end *exactly* with `]` (closing square bracket for the JSON list).
    *   Do NOT include any conversational introductions (e.g., "Here are the filtered results:"), explanations of your general process, justifications, markdown formatting like ````json ... ```` tags, or any characters whatsoever outside of the JSON list itself.
    *   The JSON should be formatted with an indent of 2 spaces (this is for readability; the most important aspect is valid, structurally compliant JSON).

    **Execution Mandate:**
    Filter the provided search results sample based *only* on the suitability criteria defined above.
    Your entire output MUST be **ONLY** the specified JSON list, starting with `[` (or `[]`) and ending with `]`.
    Ensure the JSON is well-formed (e.g., correct use of commas, quotes around keys and string values, proper nesting of objects within the list).
    Do not output any text or markdown before the opening `[` or after the closing `]`.
    """
    

# === FIRST STAGE FOCUS ANALYSIS PROMPT ===
def get_chunk_relevance_scoring_prompt(topic: str, chunk_text: str) -> str:
    return f"""
      **Your Role:** You are an AI assistant specialized in **Content Relevance Scoring for LLM Dataset Curation**. Your focus is on evaluating how well a given text **chunk** explains the fundamentals of a specific topic.

      **Your Objective:** Evaluate the provided text chunk below and assign a **percentage score (from 0 to 100)** representing the degree to which this **specific chunk** contributes to explaining the basics, fundamentals, concepts, or workings of the target topic: **"{topic}"**. The score should reflect how well this chunk, *on its own*, provides foundational, explanatory knowledge *about* "{topic}", suitable for inclusion in an LLM dataset focused on such knowledge.

      **Text Chunk to Analyze:**
      --- START OF CHUNK ---
        {chunk_text}
      --- END OF CHUNK ---

      **Your Task:** Analyze the text chunk provided above based on the scoring guidelines below and determine the most appropriate relevance score for **this chunk**.

      **Scoring Guidelines (0-100 Scale - Applied to the CHUNK):**

      *   **Score 0:** The chunk is completely off-topic from "{topic}" or mentions it only trivially (e.g., in a reference list, unrelated context within the chunk).
      *   **Score 1-29 (Very Low Relevance):** The chunk's content is primarily focused on a much broader field or a different topic. "{topic}" might be mentioned briefly within the chunk, but there is no substantial explanation *of* "{topic}" fundamentals **within this chunk**. The explanatory value *about* "{topic}" itself within this chunk is negligible.
      *   **Score 30-59 (Low to Moderate Relevance):** The chunk is related to "{topic}" (e.g., discusses the broader field or an application), and might mention concepts related to it. There might be *some minor explanatory sentences* about "{topic}" fundamentals within the chunk, but explaining "{topic}" is clearly **not the main purpose of this specific chunk**. The chunk's primary focus lies elsewhere (e.g., context setting, results, different concept).
      *   **Score 60-89 (Moderate to High Relevance):** The chunk **actively contributes** to explaining the basics, fundamentals, concepts, or workings of "{topic}". A **significant portion of the text within this chunk** is dedicated to such explanation (e.g., defining a key term related to "{topic}", explaining a core concept, describing a fundamental mechanism or algorithm step). The chunk provides clear explanatory value about "{topic}".
      *   **Score 90-100 (Very High Relevance / Ideal Chunk):** This chunk is **highly focused on and substantially explains** a core aspect of "{topic}" fundamentals. The **majority or entirety of this specific chunk** is dedicated to providing clear, foundational explanation, definition, or description related to "{topic}". It serves well as a standalone piece of explanation for that specific aspect.

      **CRITICAL - Required Output Format:**
      *   Your response MUST be **ONLY a single integer number** between 0 and 100.
      *   For example, if the score is 75, your output MUST be exactly: `75`
      *   If the score is 8, your output MUST be exactly: `8`
      *   If the score is 100, your output MUST be exactly: `100`
      *   **ABSOLUTELY NO EXTRA TEXT:** Do NOT include percentage signs (`%`), any words (like "Score:", "The relevance is:"), explanations, justifications, or any characters whatsoever outside of the single integer number.
      *   Your entire response must consist SOLELY of the digits making up the integer score.

      **Execution Mandate:**
      Based *only* on the text chunk provided above and the scoring guidelines, output the single integer score.
      Your entire response MUST be this integer and nothing else.
    """
    

# === SECOND STAGE FOCUS ANALYSIS PROMPT ===
def get_chunk_focus_scoring_prompt(topic: str, chunk_text: str) -> str:
    return f"""
        **Your Role:** You are an AI assistant specialized in **Granular Content Focus Analysis within Text Chunks**. You analyze text chunks taken from documents already known to be generally relevant to a broader field.

        **Your Objective:** Evaluate the provided **text chunk** below and assign a **percentage score (from 0 to 100)** representing the degree to which **this specific chunk's content** is dedicated to explaining the basics, fundamentals, concepts, or workings of the target topic: **"{topic}"**. The score should reflect the suitability for an LLM dataset aimed at capturing foundational, explanatory knowledge *about* "{topic}".

        **Background Context:** This document (from which the chunk is taken) has already passed an initial relevance filter, indicating it is generally related to the broader field. Your task now is a focused analysis of *this specific chunk's content*.

        **Text Chunk to Analyze:**
        --- START OF CHUNK ---
            {chunk_text}
        --- END OF CHUNK ---

        **Your Task & Required Output Format (CRITICAL):**
        1.  **Analyze Chunk Content:** Read through the text chunk. Identify the specific points, definitions, explanations, or examples presented *within this chunk*. Assess how much of this chunk's text *directly* explains "{topic}" versus how much discusses broader context, introduces other concepts, provides examples of something else, or transitions between ideas.
        2.  **Estimate Focus Percentage for the Chunk:** Based on your analysis, estimate the percentage of **this chunk's text** that directly explains "{topic}". Consider the following guidelines:
            *   **0-29%:** "{topic}" might be mentioned in the chunk, but constitutes a very small fraction of its content. The vast majority of *this chunk* explains other concepts, provides broad context, or serves as a transition.
            *   **30-59%:** "{topic}" is discussed within the chunk, perhaps with a definition or a brief explanation, but less than half of *this chunk's content* focuses on explaining it. Significant parts of the chunk might deal with related concepts, applications, or context setting.
            *   **60-89%:** Explaining "{topic}" (its fundamentals, concepts, workings) is a primary purpose of **this chunk**. The majority of the text *within this chunk* is dedicated to explaining "{topic}".
            *   **90-100%:** **This chunk** is almost entirely devoted to explaining a specific aspect of "{topic}", with minimal text spent on anything else. It functions as a focused explanatory unit for "{topic}".

        **CRITICAL - Required Output Format:**
        *   Your response MUST be **ONLY a single integer number** between 0 and 100, representing the calculated focus score **for this chunk**.
        *   For example, if the score is 82, your output MUST be exactly: `82`
        *   If the score is 5, your output MUST be exactly: `5`
        *   If the score is 95, your output MUST be exactly: `95`
        *   **ABSOLUTELY NO EXTRA TEXT:** Do NOT include percentage signs (`%`), any words (like "Focus score:", "The score is:"), explanations, justifications, or any characters whatsoever outside of the single integer number.
        *   Your entire response must consist SOLELY of the digits making up the integer score.

        **Execution Mandate:**
        Based *only* on the text chunk provided above and the scoring guidelines, output the single integer score that best reflects **this chunk's** focus on explaining the fundamentals of "{topic}".
        Your entire response MUST be this integer and nothing else.
    """


def get_clean_content_extraction_prompt(topic: str, chunk_text: str) -> str:
    
    example_topic_placeholder = "Specific Topic Name"

    return f"""
        **Your Role:** You are an AI assistant specialized in **Content Extraction and Consolidation**. Your expertise lies in identifying and extracting only the core explanatory content related to a specific topic from larger texts, while discarding irrelevant surrounding material.

        **Your Objective:** Process the provided text chunk and extract **only** the paragraphs and sections that **substantively explain the fundamentals, basics, concepts, workings, or examples of the target topic: "{topic}"**. The goal is to create a clean, consolidated text containing only the core informational content about "{topic}", suitable for an LLM dataset.

        **Target topic:** Extract content specifically explaining: **"{topic}"**

        **Text Chunk to Process:**
        --- START OF CHUNK ---
            {chunk_text}
        --- END OF CHUNK ---

        **Your Task & Required Output Format (CRITICAL):**

        1.  **Analyze and Identify Relevant Content:** Read through the `Text Chunk to Process` provided above. Identify **only** the sentences, paragraphs, or sections that contain **direct, substantial, and explanatory information specifically about "{topic}"**.
        2.  **Identify and Discard Irrelevant Content:** Explicitly **ignore and exclude** all other content, including (but not limited to): Generic introductions/conclusions, author/affiliation info, publication details, navigation elements, reference lists, bibliographies, citations, acknowledgements, figure/table captions (unless the caption itself is the core explanation), broader field discussions not directly explaining "{topic}", boilerplate text, metadata. Exclude detailed mathematical derivations and formulas, focusing only on the conceptual text explanation.
        3.  **Consolidate and Format Extracted Text:** Combine the **extracted relevant text segments** into a single, continuous text flow. Maintain paragraph breaks (double newlines `\\n\\n`) between distinct extracted paragraphs. Ensure the final text flows reasonably well and contains *only* the cleaned, topic-focused content.
        4.  **Output - Strict JSON Structure:**
            Your response MUST be **ONLY a single, valid JSON object**.
            This JSON object MUST have a single key: `"output"`.
            The value associated with the `"output"` key MUST be a **single string**.
            *   This string will contain the cleaned and consolidated text. Paragraph breaks MUST be `\\n\\n`.
            *   If NO relevant text is found, the value of `"output"` MUST be an empty string (`""`).

        *   **Strict JSON Output Examples:**
            *   Example 1: Relevant Content Found (multiple paragraphs):
                ```json
                {{
                  "output": "This is the first extracted paragraph explaining a core concept of {example_topic_placeholder}.\\n\\nThis is the second extracted paragraph giving an example related to {example_topic_placeholder}."
                }}
                ```
            *   Example 2: Relevant Content Found (single paragraph):
                ```json
                {{
                  "output": "This single paragraph contains the core explanation of {example_topic_placeholder}."
                }}
                ```
            *   Example 3: No Relevant Content Found:
                ```json
                {{
                  "output": ""
                }}
                ```

        *   **ABSOLUTELY NO EXTRA TEXT:** Your response must start *exactly* with `{{` and end *exactly* with `}}`.
        *   Do NOT include any conversational introductions, explanations of your process, justifications, markdown formatting like ````json ... ```` tags, or any characters whatsoever outside of the JSON object itself.
        *   The JSON should be formatted with an indent of 2 spaces.

        **Execution Mandate:**
        Process the provided text chunk according to all instructions above.
        Your entire output MUST be **ONLY** the specified JSON object, starting with `{{` and ending with `}}`.
        Ensure the JSON is well-formed.
        Do not output any text or markdown before the opening `{{` or after the closing `}}`.
    """
    

def get_questions_generate_prompt(chunk_text: str, num_questions_to_generate: int) -> str:

    return f"""
        **Your Role:** You are an AI assistant specialized in generating insightful, text-grounded questions.

        **Your Objective:** Based *only* on the document content provided below, you must generate **exactly {num_questions_to_generate}** relevant and insightful questions. These questions should test understanding of the main topics, key details, or potential implications discussed *within the provided document content*.
        *   Crucially, all questions generated MUST be answerable *solely* from the provided text. Do not ask questions requiring external knowledge.
        *   Each question generated should be distinct.

        **Document Content to Analyze:**
        --- START OF DOCUMENT ---
            {chunk_text}
        --- END OF DOCUMENT ---

        **CRITICAL - Required Output Format:**
        *   Your response MUST consist **ONLY** of the numbered list of questions.
        *   There MUST be **exactly {num_questions_to_generate}** questions in the list.
        *   Each question MUST start with a number, followed by a period, and then a single space (e.g., "1. ", "2. ", "3. ", etc.).
        *   Each question MUST be on a new line.
        *   The numbering MUST be sequential, starting from 1 and going up to {num_questions_to_generate}.

        *   **Strict Output Example (if {num_questions_to_generate} were, for instance, 3):**
            ```text
            1. What is the primary definition of [key term] as provided in the document?
            2. According to the text, what are two main consequences of [specific event or concept]?
            3. How does the document explain the relationship between [topic A] and [topic B]?
            ```
            (Your actual output must contain exactly {num_questions_to_generate} questions, following this line-by-line numbered format.)

        *   **ABSOLUTELY NO EXTRA TEXT OR DEVIATIONS:**
            *   Do NOT include any introductory phrases such as "Here are the questions:", "Okay, I will generate the questions:", "Generated questions:", or any similar conversational text before the first question.
            *   Do NOT include any concluding remarks, summaries, or any other text after the last character of the {num_questions_to_generate}th question.
            *   Do NOT use bullet points (e.g., "-", "*") or any format other than "Number. Question".
            *   Your entire response MUST start with "1. " (the first number, period, space of the first question) and end with the last character of the final ({num_questions_to_generate}th) question.

        **Execution Mandate:**
        Generate **exactly {num_questions_to_generate}** questions now, adhering strictly to all instructions and formatting requirements detailed above.
        Your entire output MUST be ONLY the numbered list of questions.
    """
    
    
def get_answered_questions_prompt(chunk_text: str, question: str) -> str:

    fallback_phrase = "The document does not provide an answer to this question."

    return f"""
        **Your Role:** You are an AI assistant specialized in answering questions based **strictly and solely** on the provided text document. Your primary function is information retrieval from the given context.

        **Your Objective:**
        1.  Use **only** the information explicitly available in the document content provided below to answer the given question.
        2.  You MUST NOT use any prior knowledge, assumptions, or information from outside this specific document.
        3.  Your answer should be concise, direct, and accurately reflect the information presented in the document relevant to the question.

        **Document Content to Analyze:**
        --- START OF DOCUMENT ---
            {chunk_text}
        --- END OF DOCUMENT ---

        **Question to Answer:**
            {question}

        **CRITICAL - Required Output Format and Behavior:**

        1.  **If the answer IS found in the document:**
            *   Provide a concise and direct answer derived exclusively from the document text.
            *   Avoid introductory phrases like "According to the document...", "The document states that...", or "Based on the text..." unless absolutely essential for the grammatical sense or clarity of the answer itself. The answer should directly address the question.
            *   Do not add any commentary, summarization of your process, or information beyond what is needed to directly answer the question.

        2.  **If the answer CANNOT be found in the document:**
            *   You MUST respond with the **exact and complete phrase**: "{fallback_phrase}"
            *   Your entire response in this case MUST be ONLY this phrase.
            *   **ABSOLUTELY NO VARIATIONS OR ADDITIONS:** Do not say "Unfortunately, the document...", "I could not find...", or add any apology, explanation, or any other text before or after this exact phrase.

        *   **Output Structure (after your internal processing):**
            Your final output should be the direct answer text (if found) or the exact fallback phrase (if not found).
            Example if answer found (actual answer will vary):
            `The main component described is X, which facilitates Y.`

            Example if answer NOT found (this MUST be the literal output):
            `{fallback_phrase}`

        **Execution Mandate:**
        Based **strictly and solely** on the "Document Content to Analyze" provided above, answer the "Question to Answer".
        Adhere meticulously to the output format and behavior rules, especially regarding the fallback phrase if the answer is not present.
        Your entire response will be either the direct answer or the specified fallback phrase.

        Answer:
    """
    
    

def get_check_dataset_prompt(topic: str, dataset_json_string: str) -> str:

    return f"""
        You are a meticulous AI expert specializing in refining and enhancing question-answer datasets.
        Your task is to process the following JSON dataset. The dataset is provided as a JSON string representing an array of objects, where each object contains an 'input' (question) and 'output' (answer) pair related to the topic: "{topic}".

        Your primary objectives are to meticulously review and revise EACH 'output' (answer) based on its corresponding 'input' (question) and the overall topic, adhering to the following strict guidelines:

        1.  **Ensure Accuracy and Consistency:**
            *   Verify that all answers are factually correct and directly relevant to the provided question and the overall topic: "{topic}".
            *   Identify and resolve any inconsistencies or contradictions between answers to similar or related questions. For example, if two very similar questions have drastically different answers (one being helpful, the other stating information is unavailable), reconcile them to provide the most accurate and complete response. Strive for a single, consistent truth.

        2.  **Address "Information Not Available" Issues:**
            *   If an answer states "The document does not provide an answer to this question," "The document does not explicitly list," or similar phrases indicating missing information from a *specific source document*:
                *   Attempt to provide a concise, accurate answer based on general, verifiable knowledge of the "{topic}". Do NOT speculate or invent information.
                *   If a general knowledge answer is not feasible or would be too speculative for the "{topic}", you may retain a statement about information unavailability but rephrase it to be more general and less document-specific, e.g., "Specific details for this query are not broadly available without further context." or "This information can vary greatly depending on the specific platform/context."
                *   Prioritize providing a helpful, fact-based answer if reasonably possible. If an answer cannot be found through general knowledge, state so clearly.

        3.  **Enhance Completeness and Clarity:**
            *   Expand on answers that are overly brief, cryptic, or lack sufficient detail to be truly helpful. Provide more context or explanation where it would enhance understanding, while still aiming for conciseness and relevance.
            *   Clarify any ambiguous phrasing in either questions or answers.
            *   Ensure technical terms are used correctly and, if necessary, briefly explained implicitly or explicitly within the answer if the context suggests a novice audience for the "{topic}".

        4.  **Improve Natural Language and Flow:**
            *   Ensure the language used in answers is natural, fluent, and easy to understand. Avoid overly robotic, 'bookish', or stilted phrasing.
            *   Correct any grammatical errors or awkward sentence structures.

        5.  **Handle Platform-Specific vs. General Questions:**
            *   If a question is highly specific to a particular (potentially undocumented) platform interface (e.g., "What does the yellow bar on the left do?"), and a direct answer relies on that specific interface:
                *   If the underlying concept is common (e.g., toolbars, price axes), try to provide a more generalized answer that explains the common functionality, e.g., "In many charting platforms, sidebars or specific colored elements are used to highlight toolsets, such as drawing tools or indicators."
                *   If generalization isn't truly possible and no general knowledge applies, ensure the answer is as clear as it can be, or if appropriate, use the guideline in point 2.

        6.  **Strict Output Format - Adherence is CRITICAL:**
            *   The output MUST be a single, valid JSON string.
            *   This JSON string MUST represent an ARRAY of objects.
            *   Each object in the array MUST EXACTLY mirror the input object structure, containing only two keys: "input" and "output".
            *   The "input" key's value (the question) should generally remain unchanged, but you may make minor clarifications if absolutely necessary for the answer to make sense. However, prioritize keeping the original question.
            *   The "output" key's value (the answer) MUST be your revised and improved answer.
            *   Do NOT alter the key names "input" or "output".
            *   Do NOT add any new keys to the objects.
            *   Do NOT change the order of objects within the array.
            *   The number of objects in the output array MUST be identical to the number of objects in the input array.

        **Input JSON Dataset Structure Reminder:**
        The input will be a JSON string formatted like this:
        ```json
        [
            {{"input": "Question 1 text", "output": "Original answer 1 text"}},
            {{"input": "Question 2 text", "output": "Original answer 2 text"}},
            ...
        ]
        ```

        **Actual Input JSON Dataset to Process:**
        ```json
        {dataset_json_string}
        ```

        **Your Sole Output MUST be the revised JSON dataset as a single, valid JSON string.
        It MUST start with `[` and end with `]`.
        It MUST be an array of objects, each with exactly an "input" and an "output" key.
        Do NOT include ANY introductory text, explanations, apologies, summaries, or ANY conversational elements or markdown formatting before or after the JSON output.
        ONLY the JSON string.

        Example of expected output format:
        ```json
        [
            {{"input": "Question 1 text (or slightly clarified)", "output": "Revised, improved answer 1 text"}},
            {{"input": "Question 2 text (or slightly clarified)", "output": "Revised, improved answer 2 text"}},
            ...
        ]
        ```
    """