from langchain_openai import ChatOpenAI
import httpx

# Disable SSL verification - usually not recommended for production use
client = httpx.Client(verify=False)

# Initialize the LLM
llm = ChatOpenAI(
    base_url="https://genailab.tcs.in",  # Custom base URL for inference endpoint
    model="azure_ai/genailab-maas-DeepSeek-V3-0324",  # Model identifier
    api_key="sk-F2PlNEIXzjK7VmAmN7aSSg",  # Dummy/placeholder API key
    http_client=client
)

# Invoke the model with a prompt
response = llm.invoke("write a story on fantasy")

# Print the output
print(response.content)



