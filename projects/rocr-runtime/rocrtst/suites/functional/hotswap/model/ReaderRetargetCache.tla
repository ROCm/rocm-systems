-------------------------- MODULE ReaderRetargetCache --------------------------
EXTENDS Integers, Naturals, TLC

CONSTANTS Clients, Keys, Results, SuccessResults,
          NoClient, NoKey, NoRecord, MaxGenerations

ASSUME /\ Clients # {}
       /\ Keys # {}
       /\ SuccessResults \subseteq Results
       /\ NoClient \notin Clients
       /\ NoKey \notin Keys
       /\ MaxGenerations \in Nat \ {0}

CacheStates == {"Absent", "Computing", "Ready"}
ClientStates == {"Idle", "Producing", "Waiting", "Done"}
GenerationRange == 1..MaxGenerations
ResultRecords == [key : Keys, generation : GenerationRange, result : Results]

VARIABLES cacheState, leader, waiters, readyResult, generation,
          clientState, requestKey, joinedGeneration, pendingResult,
          observedResult, producerCount

vars == <<cacheState, leader, waiters, readyResult, generation,
          clientState, requestKey, joinedGeneration, pendingResult,
          observedResult, producerCount>>

Init ==
  /\ cacheState = [key \in Keys |-> "Absent"]
  /\ leader = [key \in Keys |-> NoClient]
  /\ waiters = [key \in Keys |-> {}]
  /\ readyResult = [key \in Keys |-> NoRecord]
  /\ generation = [key \in Keys |-> 0]
  /\ clientState = [client \in Clients |-> "Idle"]
  /\ requestKey = [client \in Clients |-> NoKey]
  /\ joinedGeneration = [client \in Clients |-> 0]
  /\ pendingResult = [client \in Clients |-> NoRecord]
  /\ observedResult = [client \in Clients |-> NoRecord]
  /\ producerCount = [key \in Keys |-> 0]

StartReady(client, key) ==
  /\ clientState[client] = "Idle"
  /\ cacheState[key] = "Ready"
  /\ clientState' = [clientState EXCEPT ![client] = "Done"]
  /\ requestKey' = [requestKey EXCEPT ![client] = key]
  /\ joinedGeneration' = [joinedGeneration EXCEPT ![client] = generation[key]]
  /\ observedResult' = [observedResult EXCEPT ![client] = readyResult[key]]
  /\ UNCHANGED <<cacheState, leader, waiters, readyResult, generation,
                  pendingResult, producerCount>>

StartProducer(client, key) ==
  /\ clientState[client] = "Idle"
  /\ cacheState[key] = "Absent"
  /\ generation[key] < MaxGenerations
  /\ cacheState' = [cacheState EXCEPT ![key] = "Computing"]
  /\ leader' = [leader EXCEPT ![key] = client]
  /\ generation' = [generation EXCEPT ![key] = @ + 1]
  /\ clientState' = [clientState EXCEPT ![client] = "Producing"]
  /\ requestKey' = [requestKey EXCEPT ![client] = key]
  /\ joinedGeneration' = [joinedGeneration EXCEPT ![client] = generation[key] + 1]
  /\ observedResult' = [observedResult EXCEPT ![client] = NoRecord]
  /\ producerCount' = [producerCount EXCEPT ![key] = @ + 1]
  /\ UNCHANGED <<waiters, readyResult, pendingResult>>

JoinFlight(client, key) ==
  /\ clientState[client] = "Idle"
  /\ cacheState[key] = "Computing"
  /\ client # leader[key]
  /\ waiters' = [waiters EXCEPT ![key] = @ \cup {client}]
  /\ clientState' = [clientState EXCEPT ![client] = "Waiting"]
  /\ requestKey' = [requestKey EXCEPT ![client] = key]
  /\ joinedGeneration' = [joinedGeneration EXCEPT ![client] = generation[key]]
  /\ observedResult' = [observedResult EXCEPT ![client] = NoRecord]
  /\ UNCHANGED <<cacheState, leader, readyResult, generation,
                  pendingResult, producerCount>>

Publish(client, key, result) ==
  LET published == [key |-> key, generation |-> generation[key], result |-> result]
  IN
  /\ cacheState[key] = "Computing"
  /\ leader[key] = client
  /\ clientState[client] = "Producing"
  /\ cacheState' = [cacheState EXCEPT
                      ![key] = IF result \in SuccessResults THEN "Ready" ELSE "Absent"]
  /\ readyResult' = [readyResult EXCEPT
                       ![key] = IF result \in SuccessResults THEN published ELSE NoRecord]
  /\ leader' = [leader EXCEPT ![key] = NoClient]
  /\ waiters' = [waiters EXCEPT ![key] = {}]
  /\ pendingResult' =
       [other \in Clients |->
          IF other \in waiters[key] THEN published ELSE pendingResult[other]]
  /\ clientState' = [clientState EXCEPT ![client] = "Done"]
  /\ observedResult' = [observedResult EXCEPT ![client] = published]
  /\ UNCHANGED <<generation, requestKey, joinedGeneration, producerCount>>

WakeWaiter(client) ==
  /\ clientState[client] = "Waiting"
  /\ pendingResult[client] # NoRecord
  /\ clientState' = [clientState EXCEPT ![client] = "Done"]
  /\ observedResult' = [observedResult EXCEPT ![client] = pendingResult[client]]
  /\ pendingResult' = [pendingResult EXCEPT ![client] = NoRecord]
  /\ UNCHANGED <<cacheState, leader, waiters, readyResult, generation,
                  requestKey, joinedGeneration, producerCount>>

ResetClient(client) ==
  /\ clientState[client] = "Done"
  /\ clientState' = [clientState EXCEPT ![client] = "Idle"]
  /\ requestKey' = [requestKey EXCEPT ![client] = NoKey]
  /\ joinedGeneration' = [joinedGeneration EXCEPT ![client] = 0]
  /\ observedResult' = [observedResult EXCEPT ![client] = NoRecord]
  /\ UNCHANGED <<cacheState, leader, waiters, readyResult, generation,
                  pendingResult, producerCount>>

ExpireWeakEntry(key) ==
  /\ cacheState[key] = "Ready"
  /\ cacheState' = [cacheState EXCEPT ![key] = "Absent"]
  /\ readyResult' = [readyResult EXCEPT ![key] = NoRecord]
  /\ UNCHANGED <<leader, waiters, generation, clientState, requestKey,
                  joinedGeneration, pendingResult, observedResult, producerCount>>

Next ==
  \/ \E client \in Clients, key \in Keys : StartReady(client, key)
  \/ \E client \in Clients, key \in Keys : StartProducer(client, key)
  \/ \E client \in Clients, key \in Keys : JoinFlight(client, key)
  \/ \E client \in Clients, key \in Keys, result \in Results : Publish(client, key, result)
  \/ \E client \in Clients : WakeWaiter(client)
  \/ \E client \in Clients : ResetClient(client)
  \/ \E key \in Keys : ExpireWeakEntry(key)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ cacheState \in [Keys -> CacheStates]
  /\ leader \in [Keys -> Clients \cup {NoClient}]
  /\ waiters \in [Keys -> SUBSET Clients]
  /\ readyResult \in [Keys -> ResultRecords \cup {NoRecord}]
  /\ generation \in [Keys -> 0..MaxGenerations]
  /\ clientState \in [Clients -> ClientStates]
  /\ requestKey \in [Clients -> Keys \cup {NoKey}]
  /\ joinedGeneration \in [Clients -> 0..MaxGenerations]
  /\ pendingResult \in [Clients -> ResultRecords \cup {NoRecord}]
  /\ observedResult \in [Clients -> ResultRecords \cup {NoRecord}]
  /\ producerCount \in [Keys -> Nat]

OneLeaderPerFlight ==
  \A key \in Keys : (cacheState[key] = "Computing") <=> (leader[key] \in Clients)

LeaderIsProducing ==
  \A key \in Keys :
    leader[key] \in Clients =>
      /\ clientState[leader[key]] = "Producing"
      /\ requestKey[leader[key]] = key
      /\ joinedGeneration[leader[key]] = generation[key]

WaitersMatchFlight ==
  \A key \in Keys :
    \A client \in waiters[key] :
      /\ cacheState[key] = "Computing"
      /\ clientState[client] = "Waiting"
      /\ requestKey[client] = key
      /\ joinedGeneration[client] = generation[key]

WaitingHasFlightOrPublishedResult ==
  \A client \in Clients :
    clientState[client] = "Waiting" =>
      \/ pendingResult[client] \in ResultRecords
      \/ \E key \in Keys : client \in waiters[key]

ReadyHasSuccessfulResult ==
  \A key \in Keys :
    cacheState[key] = "Ready" =>
      /\ readyResult[key] \in ResultRecords
      /\ readyResult[key].key = key
      /\ readyResult[key].generation = generation[key]
      /\ readyResult[key].result \in SuccessResults

ObservedGenerationAgreement ==
  \A left, right \in ResultRecords :
    /\ (\E client \in Clients : observedResult[client] = left)
    /\ (\E client \in Clients : observedResult[client] = right)
    /\ left.key = right.key
    /\ left.generation = right.generation
    => left.result = right.result

=============================================================================
