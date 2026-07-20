------------------------- MODULE ContentRetargetCache -------------------------
EXTENDS Integers, Naturals, TLC

CONSTANTS Clients, Transforms, Contents, Buckets, Readers, BucketOf(_),
          Results, SuccessResults, NoClient, NoKey, NoReader, NoRecord,
          MaxGenerations

ExactKeys == Transforms \X Contents
BucketKey(key) == <<key[1], BucketOf(key[2])>>
ForcedCollisionBucket(content) == CHOOSE bucket \in Buckets : TRUE

ASSUME /\ Clients # {}
       /\ Transforms # {}
       /\ Contents # {}
       /\ Buckets # {}
       /\ Readers # {}
       /\ \A content \in Contents : BucketOf(content) \in Buckets
       /\ SuccessResults \subseteq Results
       /\ NoClient \notin Clients
       /\ NoKey \notin ExactKeys
       /\ NoReader \notin Readers
       /\ MaxGenerations \in Nat \ {0}

CacheStates == {"Absent", "Computing", "Ready"}
ClientStates == {"Idle", "Producing", "Waiting", "Done"}
GenerationRange == 1..MaxGenerations
ResultRecords == [key : ExactKeys, generation : GenerationRange, result : Results]

VARIABLES cacheState, leader, waiters, readyResult, generation,
          clientState, requestKey, requestReader, joinedGeneration,
          pendingResult, observedResult, producerCount

vars == <<cacheState, leader, waiters, readyResult, generation,
          clientState, requestKey, requestReader, joinedGeneration,
          pendingResult, observedResult, producerCount>>

Init ==
  /\ cacheState = [key \in ExactKeys |-> "Absent"]
  /\ leader = [key \in ExactKeys |-> NoClient]
  /\ waiters = [key \in ExactKeys |-> {}]
  /\ readyResult = [key \in ExactKeys |-> NoRecord]
  /\ generation = [key \in ExactKeys |-> 0]
  /\ clientState = [client \in Clients |-> "Idle"]
  /\ requestKey = [client \in Clients |-> NoKey]
  /\ requestReader = [client \in Clients |-> NoReader]
  /\ joinedGeneration = [client \in Clients |-> 0]
  /\ pendingResult = [client \in Clients |-> NoRecord]
  /\ observedResult = [client \in Clients |-> NoRecord]
  /\ producerCount = [key \in ExactKeys |-> 0]

StartReady(client, key, reader) ==
  /\ clientState[client] = "Idle"
  /\ cacheState[key] = "Ready"
  /\ clientState' = [clientState EXCEPT ![client] = "Done"]
  /\ requestKey' = [requestKey EXCEPT ![client] = key]
  /\ requestReader' = [requestReader EXCEPT ![client] = reader]
  /\ joinedGeneration' = [joinedGeneration EXCEPT ![client] = generation[key]]
  /\ observedResult' = [observedResult EXCEPT ![client] = readyResult[key]]
  /\ UNCHANGED <<cacheState, leader, waiters, readyResult, generation,
                  pendingResult, producerCount>>

StartProducer(client, key, reader) ==
  /\ clientState[client] = "Idle"
  /\ cacheState[key] = "Absent"
  /\ generation[key] < MaxGenerations
  /\ cacheState' = [cacheState EXCEPT ![key] = "Computing"]
  /\ leader' = [leader EXCEPT ![key] = client]
  /\ generation' = [generation EXCEPT ![key] = @ + 1]
  /\ clientState' = [clientState EXCEPT ![client] = "Producing"]
  /\ requestKey' = [requestKey EXCEPT ![client] = key]
  /\ requestReader' = [requestReader EXCEPT ![client] = reader]
  /\ joinedGeneration' = [joinedGeneration EXCEPT ![client] = generation[key] + 1]
  /\ observedResult' = [observedResult EXCEPT ![client] = NoRecord]
  /\ producerCount' = [producerCount EXCEPT ![key] = @ + 1]
  /\ UNCHANGED <<waiters, readyResult, pendingResult>>

JoinFlight(client, key, reader) ==
  /\ clientState[client] = "Idle"
  /\ cacheState[key] = "Computing"
  /\ client # leader[key]
  /\ waiters' = [waiters EXCEPT ![key] = @ \cup {client}]
  /\ clientState' = [clientState EXCEPT ![client] = "Waiting"]
  /\ requestKey' = [requestKey EXCEPT ![client] = key]
  /\ requestReader' = [requestReader EXCEPT ![client] = reader]
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
  /\ UNCHANGED <<generation, requestKey, requestReader, joinedGeneration,
                  producerCount>>

WakeWaiter(client) ==
  /\ clientState[client] = "Waiting"
  /\ pendingResult[client] # NoRecord
  /\ clientState' = [clientState EXCEPT ![client] = "Done"]
  /\ observedResult' = [observedResult EXCEPT ![client] = pendingResult[client]]
  /\ pendingResult' = [pendingResult EXCEPT ![client] = NoRecord]
  /\ UNCHANGED <<cacheState, leader, waiters, readyResult, generation,
                  requestKey, requestReader, joinedGeneration, producerCount>>

ResetClient(client) ==
  /\ clientState[client] = "Done"
  /\ clientState' = [clientState EXCEPT ![client] = "Idle"]
  /\ requestKey' = [requestKey EXCEPT ![client] = NoKey]
  /\ requestReader' = [requestReader EXCEPT ![client] = NoReader]
  /\ joinedGeneration' = [joinedGeneration EXCEPT ![client] = 0]
  /\ observedResult' = [observedResult EXCEPT ![client] = NoRecord]
  /\ UNCHANGED <<cacheState, leader, waiters, readyResult, generation,
                  pendingResult, producerCount>>

ExpireWeakEntry(key) ==
  /\ cacheState[key] = "Ready"
  /\ cacheState' = [cacheState EXCEPT ![key] = "Absent"]
  /\ readyResult' = [readyResult EXCEPT ![key] = NoRecord]
  /\ UNCHANGED <<leader, waiters, generation, clientState, requestKey,
                  requestReader, joinedGeneration, pendingResult,
                  observedResult, producerCount>>

Next ==
  \/ \E client \in Clients, key \in ExactKeys, reader \in Readers :
       StartReady(client, key, reader)
  \/ \E client \in Clients, key \in ExactKeys, reader \in Readers :
       StartProducer(client, key, reader)
  \/ \E client \in Clients, key \in ExactKeys, reader \in Readers :
       JoinFlight(client, key, reader)
  \/ \E client \in Clients, key \in ExactKeys, result \in Results :
       Publish(client, key, result)
  \/ \E client \in Clients : WakeWaiter(client)
  \/ \E client \in Clients : ResetClient(client)
  \/ \E key \in ExactKeys : ExpireWeakEntry(key)

Spec == Init /\ [][Next]_vars

TypeOK ==
  /\ cacheState \in [ExactKeys -> CacheStates]
  /\ leader \in [ExactKeys -> Clients \cup {NoClient}]
  /\ waiters \in [ExactKeys -> SUBSET Clients]
  /\ readyResult \in [ExactKeys -> ResultRecords \cup {NoRecord}]
  /\ generation \in [ExactKeys -> 0..MaxGenerations]
  /\ clientState \in [Clients -> ClientStates]
  /\ requestKey \in [Clients -> ExactKeys \cup {NoKey}]
  /\ requestReader \in [Clients -> Readers \cup {NoReader}]
  /\ joinedGeneration \in [Clients -> 0..MaxGenerations]
  /\ pendingResult \in [Clients -> ResultRecords \cup {NoRecord}]
  /\ observedResult \in [Clients -> ResultRecords \cup {NoRecord}]
  /\ producerCount \in [ExactKeys -> Nat]

OneLeaderPerFlight ==
  \A key \in ExactKeys : (cacheState[key] = "Computing") <=> (leader[key] \in Clients)

LeaderIsProducing ==
  \A key \in ExactKeys :
    leader[key] \in Clients =>
      /\ clientState[leader[key]] = "Producing"
      /\ requestKey[leader[key]] = key
      /\ joinedGeneration[leader[key]] = generation[key]

WaitersMatchFlight ==
  \A key \in ExactKeys :
    \A client \in waiters[key] :
      /\ cacheState[key] = "Computing"
      /\ clientState[client] = "Waiting"
      /\ requestKey[client] = key
      /\ joinedGeneration[client] = generation[key]

WaitingHasFlightOrPublishedResult ==
  \A client \in Clients :
    clientState[client] = "Waiting" =>
      \/ pendingResult[client] \in ResultRecords
      \/ \E key \in ExactKeys : client \in waiters[key]

ReadyHasSuccessfulResult ==
  \A key \in ExactKeys :
    cacheState[key] = "Ready" =>
      /\ readyResult[key] \in ResultRecords
      /\ readyResult[key].key = key
      /\ readyResult[key].generation = generation[key]
      /\ readyResult[key].result \in SuccessResults

ObservedExactKey ==
  \A client \in Clients :
    observedResult[client] \in ResultRecords =>
      observedResult[client].key = requestKey[client]

CollidingContentsRemainDistinct ==
  \A left, right \in ExactKeys :
    /\ left # right
    /\ BucketKey(left) = BucketKey(right)
    /\ cacheState[left] = "Ready"
    /\ cacheState[right] = "Ready"
    => readyResult[left].key # readyResult[right].key

ObservedGenerationAgreement ==
  \A left, right \in ResultRecords :
    /\ (\E client \in Clients : observedResult[client] = left)
    /\ (\E client \in Clients : observedResult[client] = right)
    /\ left.key = right.key
    /\ left.generation = right.generation
    => left.result = right.result

=============================================================================
