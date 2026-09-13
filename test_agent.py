from agent import run_agent


print("\n================================")
print("      TECHNOVA AI AGENT")
print("================================")


question = input(
    "\nCustomer: "
)


response = run_agent(
    question
)


print(
    "\n🤖 Agent:"
)

print(response)