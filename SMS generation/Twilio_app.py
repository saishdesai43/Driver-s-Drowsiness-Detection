from twilio.rest import Client
# pz8ppc0vq_b0HwTdcw2DcWLaVdw4egazaZBy2mwP
account_sid = "AC902f5099931ee53d45992de15cf56548"
auth_token = "5f49e8e98ae3e775e447149be80fd8bb"

client = Client(account_sid, auth_token)

message = client.messages.create(
    body="Wear mask properly. Otherwise, Strict action will be taken.",
    from_="+17122145253",
    to="+919689417408"
)
print(message.sid)
