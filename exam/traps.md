# Trap exam

Idea credited to Delvefall (github.com/Ballchinian/Delvefall); items here are our own. T001 and T002 come from the owner's review of the app. T003 to T051 were proposed by the text rules in `src/mtgemb/exam.py` and accepted by the owner on 2026-10-18. The rules used the same Tagger tags the model trains on, so the exam is not independent of the training labels.
Rule: **same mechanism, flexible parameters.** Numbers, colours and riders are forgivable;
a different or flipped mechanism is not.

Each item: an anchor (a whole card, or `line N` of it), a true **match** and a **trap**.
A model passes when it scores the match above the trap. Only the user changes `proposed` to
`accepted` or `rejected` in an item's heading. Only accepted items are scored. Never edit
ids or card ids; to change an item, reject it and add a new one with the next id.
Every card named here (and its family) is kept out of new auxiliary training rows (line-tag and synthetic rows). Fit exam cards are still in the original card pairs, so the exam is not held out.
Sealed cards never appear.

## T001 · user · accepted
anchor: ae120613-97d6-4393-b39d-c3e6c076f5d6 | Deflecting Swat (dev) | If you control a commander, you may cast this spell without paying its mana cost. / You may choose new targets for target spell or ability.
match: 74a7d880-5a49-412b-af4d-0b4cdb8453ee | Redirect (fit) | You may choose new targets for target spell.
trap: d0a31db0-69db-405c-99a3-945617900c54 | Obscuring Haze (fit) | If you control a commander, you may cast this spell without paying its mana cost. / Prevent all damage that would be dealt this turn by creatures your opponents control.
note: neighbours shared the commander free-cast clause, not the redirect effect; from the user's app review, 2026-10-18

## T002 · user · accepted
anchor: ea7de96f-db69-421d-a951-ffaeac140379 line 2 | Bruenor Battlehammer (fit) | You may pay {0} rather than pay the equip cost of the first equip ability you activate each turn.
match: 4df579df-fdbb-418a-af2d-709e140ee569 | Forge Anew (fit) | When this enchantment enters, return target Equipment card from your graveyard to the battlefield. / During your turn, you may activate equip abilities any time you could cast an instant. / You may pay {0} rather than pay the equip cost of the first equip ability you activate during each of your turns.
trap: 2a956224-8bdb-40d8-8405-2aa29eb228e0 | Golem-Skin Gauntlets (fit) | Equipped creature gets +1/+0 for each Equipment attached to it. / Equip {2}
note: neighbours were about Equipment counts, not free equip costs; from the user's app review, 2026-10-18

## T003 · rider · accepted
anchor: 02a8b849-e708-427c-92fe-9a41bb0ec7fe | Friendly Teddy (dev) | When this creature dies, each player draws a card.
match: 6d42dff5-97ec-4768-b112-84a3584d0b87 | Vision Skeins (fit) | Each player draws two cards.
trap: cf765efe-884c-48e2-9edb-9d45cf2756dd | Lake-town Lookout (fit) | When this creature dies, recruit.

## T004 · rider · accepted
anchor: 0c6ce7c9-316b-4171-a8fa-ca26af1eecb7 | Dark Tutelage (dev) | At the beginning of your upkeep, reveal the top card of your library and put that card into your hand. You lose life equal to its mana value.
match: 910036e7-ae27-4adb-a314-d9760bce1eb0 | Pain Seer (fit) | Inspired — Whenever this creature becomes untapped, reveal the top card of your library and put that card into your hand. You lose life equal to that card's mana value.
trap: bbf90463-280a-413e-b8a3-4577dcf11e54 | Killer Instinct (fit) | At the beginning of your upkeep, reveal the top card of your library. If it's a creature card, put it onto the battlefield. That creature gains haste until end of turn. Sacrifice it at the beginning of the next end step.

## T005 · rider · accepted
anchor: 0cbe2a00-4303-4ade-98b5-4172d97f15fb | Boilerbilges Ripper (dev) | When this creature enters, you may sacrifice another creature or enchantment. When you do, this creature deals 2 damage to any target.
match: cdd8d7cf-28f2-4a4d-a373-40d48bff615b | Last-Ditch Effort (fit) | Sacrifice any number of creatures. Last-Ditch Effort deals that much damage to any target.
trap: 0618a217-2a6e-4fa1-bfe6-89c6612a594d | K'un-Lun Warrior (fit) | When this creature enters, you may sacrifice an artifact or discard a card. If you do, draw a card.

## T006 · rider · accepted
anchor: 10064a1d-66f4-4018-94c2-c727e7838850 | Demonic Rising (dev) | At the beginning of your end step, if you control exactly one creature, create a 5/5 black Demon creature token with flying.
match: 2a26dee4-b4b6-489d-99e5-6dbd8d19419d | Security Detail (fit) | {W}{W}: Create a 1/1 white Soldier creature token. Activate only if you control no creatures and only once each turn.
trap: 610d006a-3ceb-4136-b92b-a500c74d06b0 | Frontline War-Rager (fit) | At the beginning of your end step, if you control two or more tapped creatures, put a +1/+1 counter on this creature.

## T007 · rider · accepted
anchor: 11ad664e-36d3-4d5b-8a87-59ea715877e0 | Vindictive Vampire (dev) | Whenever another creature you control dies, this creature deals 1 damage to each opponent and you gain 1 life.
match: b2f2645f-5f74-456a-bd02-83169d8b8a7e | Vraan, Executioner Thane (fit) | Whenever one or more other creatures you control die, each opponent loses 2 life and you gain 2 life. This ability triggers only once each turn.
trap: 05d0b904-a680-4f2f-aaa1-e227a7fe7e79 | Unruly Mob (fit) | Whenever another creature you control dies, put a +1/+1 counter on this creature.

## T008 · rider · accepted
anchor: 17b29350-4f37-4552-8192-4856b15345f9 | Smuggler's Share (dev) | At the beginning of each end step, draw a card for each opponent who drew two or more cards this turn, then create a Treasure token for each opponent who had two or more lands enter the battlefield under their control this turn.
match: 63fd2a57-7a47-4e07-947c-f4e9da7ee538 | Mind's Eye (fit) | Whenever an opponent draws a card, you may pay {1}. If you do, draw a card.
trap: ff5502d2-37f5-4ad5-bee2-3a03ec973268 | Lasting Tarfire (fit) | At the beginning of each end step, if you put a counter on a creature this turn, this enchantment deals 2 damage to each opponent.

## T009 · rider · accepted
anchor: 2d09e689-c163-48c5-af5d-766f110d041e | Spined Tyrranax (dev) | At the beginning of combat on your turn, you may pay {2}{G}. When you do, put a +1/+1 counter on target creature. That creature gains trample until end of turn.
match: 02acf8b2-6108-4496-9803-7199ca7e199a | Ent-Draught Basin (fit) | {X}, {T}: Put a +1/+1 counter on target creature with power X. Activate only as a sorcery.
trap: ad70144e-16e4-4604-9e6d-cb287f67ddb5 | Eidolon of Inspiration (fit) | At the beginning of combat on your turn, target creature you control gets +2/+0 until end of turn.

## T010 · rider · accepted
anchor: 4cdbc466-42fc-471f-beab-397caec18101 | Vengeful Bloodwitch (dev) | Whenever this creature or another creature you control dies, target opponent loses 1 life and you gain 1 life.
match: 310f141c-7f37-4729-aed6-dd9c09db448d | Blood Artist (fit) | Whenever this creature or another creature dies, target player loses 1 life and you gain 1 life.
trap: 594a57f6-7827-41c7-aa37-00b498720b16 | Warteye Witch (fit) | Whenever this creature or another creature you control dies, scry 1.

## T011 · rider · accepted
anchor: 56c1a548-c80c-432b-bf92-d4ef6117b298 | Dimir Cutpurse (dev) | Whenever this creature deals combat damage to a player, that player discards a card and you draw a card.
match: 8a05ec32-7b0c-4f23-a4f7-413301c2a70a | Coastal Piracy (fit) | Whenever a creature you control deals combat damage to an opponent, you may draw a card.
trap: 54af9e23-459e-470d-9b0e-e91a0fa45c98 | Wandering Champion (fit) | Whenever this creature deals combat damage to a player, if you control a blue or red permanent, you may discard a card. If you do, draw a card.

## T012 · rider · accepted
anchor: 68f81fcf-2dcf-41b9-ad7c-588149feb12a | Antagonism (dev) | At the beginning of each player's end step, this enchantment deals 2 damage to that player unless one of their opponents was dealt damage this turn.
match: 77275fc5-5f70-4d7f-938e-f32b5c19c532 | Rumbling Slum (fit) | At the beginning of your upkeep, this creature deals 1 damage to each player.
trap: 53bfe54e-9b97-4113-97d9-3c5dbc85e7cf | Akal Pakal, First Among Equals (fit) | At the beginning of each player's end step, if an artifact entered the battlefield under your control this turn, look at the top two cards of your library. Put one of them into your hand and the other into your graveyard.

## T013 · rider · accepted
anchor: 7182e6f4-6caf-474a-8759-13c56f391bef | Sigil Captain (dev) | Whenever a creature you control enters, if that creature is 1/1, put two +1/+1 counters on it.
match: 1e0cdff3-3ec5-41fc-8053-f072bae156b3 | Good-Fortune Unicorn (fit) | Whenever another creature you control enters, put a +1/+1 counter on that creature.
trap: 4a782bf9-4051-4613-8852-33b0d85a0edd | Ajani's Welcome (fit) | Whenever a creature you control enters, you gain 1 life.

## T014 · rider · accepted
anchor: 83c61083-b357-4775-a855-5a9013cc2bdf | Verdant Kraken (dev) | At the beginning of each player's upkeep, you create a 3/3 green Forest Tentacle land creature token.
match: 7a21ea22-3cd7-4c11-8895-5943c0d93a0d | Verdant Force (fit) | At the beginning of each upkeep, create a 1/1 green Saproling creature token.
trap: 91edef61-2487-405c-a3ec-67a814dfeff2 | Spiritual Sanctuary (fit) | At the beginning of each player's upkeep, if that player controls a Plains, they gain 1 life.

## T015 · rider · accepted
anchor: 86a7f685-05d0-4167-b5bb-8f6943c23087 | Zhentarim Bandit (dev) | Whenever this creature attacks, you may pay 1 life. If you do, create a Treasure token.
match: ace86e56-efde-4eb7-8815-71456a4c3abe | Warren Soultrader (fit) | Pay 1 life, Sacrifice another creature: Create a Treasure token.
trap: af029853-cfb0-403b-af51-141ba02ae2e4 | Mtenda Lion (fit) | Whenever this creature attacks, defending player may pay {U}. If that player does, prevent all combat damage that would be dealt by this creature this turn.

## T016 · rider · accepted
anchor: 99f8805f-6618-4838-a4e7-a3660e3b087f | Timesifter (dev) | At the beginning of each upkeep, each player exiles the top card of their library. The player who exiled the card with the greatest mana value takes an extra turn after this one. If two or more players' cards are tied for greatest, the tied players repeat this process until the tie is broken.
match: 9567d963-4815-466b-a319-6d138b6a0cd2 | Triple Triad (fit) | At the beginning of your upkeep, each player exiles the top card of their library. Until end of turn, you may play the card you own exiled this way and each other card exiled this way with lesser mana value than it without paying their mana costs.
trap: a375b935-c02b-44bc-a942-37c5de58b1e4 | Séance (fit) | At the beginning of each upkeep, you may exile target creature card from your graveyard. If you do, create a token that's a copy of that card, except it's a Spirit in addition to its other types. Exile it at the beginning of the next end step.

## T017 · rider · accepted
anchor: dcacb717-4ca4-47c0-9b23-26e4eb4ffdc8 | Ebony Owl Netsuke (dev) | At the beginning of each opponent's upkeep, if that player has seven or more cards in hand, this artifact deals 4 damage to that player.
match: 81e6ade9-a38f-4655-b09c-9ca41a11face | Breath of Malfegor (fit) | Breath of Malfegor deals 5 damage to each opponent.
trap: a5a2d17d-5759-46fb-8e44-694bd3b72f5d | Shapeshifter's Marrow (fit) | At the beginning of each opponent's upkeep, that player reveals the top card of their library. If it's a creature card, the player puts the card into their graveyard and this enchantment becomes a copy of that card.

## T018 · rider · accepted
anchor: 045a9d3d-c20d-427f-a77b-0bffc6f55526 | Soul's Attendant (fit) | Whenever another creature enters, you may gain 1 life.
match: 0109432d-5a2a-456f-ad39-b75cb4c73420 | Bogwater Lumaret (fit) | Whenever this creature or another creature you control enters, you gain 1 life.
trap: 831e9f71-9664-4dbb-94f5-935d7cf95d77 | Saltskitter (fit) | Whenever another creature enters, exile this creature. Return this creature to the battlefield under its owner's control at the beginning of the next end step.

## T019 · rider · accepted
anchor: 4920492e-c210-4d99-ace9-184d65e697f5 | Kashi-Tribe Warriors (fit) | Whenever this creature deals combat damage to a creature, tap that creature and it doesn't untap during its controller's next untap step.
match: 74224cce-c2ce-4456-8a97-3abbbb199385 | Rowdy Snowballers (fit) | When this creature enters, tap target creature an opponent controls and put a stun counter on it.
trap: d287c11e-0684-42ed-abdb-758852ed0502 | Plague Fiend (dev) | Whenever this creature deals combat damage to a creature, destroy that creature unless its controller pays {2}.

## T020 · rider · accepted
anchor: 692a6833-3014-42f6-b1ad-333bb3292c65 | Thought Reflection (fit) | If you would draw a card, draw two cards instead.
match: 1c9e1f75-73f0-4846-b53f-458a0984b1bb | Teferi's Ageless Insight (fit) | If you would draw a card except the first one you draw in each of your draw steps, draw two cards instead.
trap: eccc9a54-2b56-4a02-9926-258d5b2e25fb | Abundance (fit) | If you would draw a card, you may instead choose land or nonland and reveal cards from the top of your library until you reveal a card of the chosen kind. Put that card into your hand and put all other cards revealed this way on the bottom of your library in any order.

## T021 · rider · accepted
anchor: 9243bd95-5467-490d-9bd2-4d5ce1ebc589 | Ghosts of the Innocent (fit) | If a source would deal damage to a permanent or player, it deals half that damage, rounded down, to that permanent or player instead.
match: ed854708-5a65-4293-bebc-d7c639407ba5 | Kitsune Palliator (dev) | {T}: Prevent the next 1 damage that would be dealt to each creature and each player this turn.
trap: 68715465-6cf9-4006-87e9-31f227fe9ed3 | Furnace of Rath (fit) | If a source would deal damage to a permanent or player, it deals double that damage to that permanent or player instead.

## T022 · mechanism · accepted
anchor: 13e8d063-eece-4f38-83f6-02403130defa | Brazen Freebooter (dev) | When this creature enters, create a Treasure token.
match: ba7c1062-f777-4ba9-84df-ec340a9ec514 | Patient Naturalist (fit) | When this creature enters, mill three cards. Put a land card from among the milled cards into your hand. If you can't, create a Treasure token.
trap: 7d48ad95-d428-4e2d-b697-56907c3ce579 | Dire Fleet Hoarder (fit) | When this creature dies, create a Treasure token.

## T023 · mechanism · accepted
anchor: 4e964f31-c764-4af5-8c27-a46fd721fab4 | Springmane Cervin (dev) | When this creature enters, you gain 2 life.
match: 05c5752a-6ccb-45ff-bb85-5f89efd5944b | Hill Giant Herdgorger (fit) | When this creature enters, you gain 3 life.
trap: 6358ee1c-3577-4196-bae4-29a3c160a66a | Toxic Abomination (fit) | When this creature enters, you lose 2 life.

## T024 · mechanism · accepted
anchor: 5bebba2d-867f-4ed1-a308-904fac32c8b6 | Feral Prowler (dev) | When this creature dies, draw a card.
match: 070bd1e8-6c61-4b0e-b179-e375b843410f | Youthful Scholar (fit) | When this creature dies, draw two cards.
trap: 8c07878f-348c-4fd9-955f-87e975ed61fa | Exiled Boggart (fit) | When this creature dies, discard a card.

## T025 · mechanism · accepted
anchor: c0406e70-8131-4e13-b1d5-4e943ad296b8 | Stand Firm (dev) | Target creature gets +1/+1 until end of turn. Scry 2.
match: 7d07821e-d34b-4441-a137-836ffce7127b | Samut's Sprint (fit) | Target creature gets +2/+1 and gains haste until end of turn. Scry 1.
trap: 4f6dbd90-f7fe-4adf-a00f-27573c0bd5c4 | Lose Hope (fit) | Target creature gets -1/-1 until end of turn. Scry 2.

## T026 · mechanism · accepted
anchor: 046b6989-5b56-4b52-a655-109b1eb26a93 | Immolating Glare (fit) | Destroy target attacking creature.
match: 28ea1f58-3310-4cf1-a834-40f2cc3b6898 | Terashi's Verdict (dev) | Destroy target attacking creature with power 3 or less.
trap: 683ec6d2-4ea7-46a7-911e-12c83e4c4a06 | Vanquish (fit) | Destroy target blocking creature.

## T027 · mechanism · accepted
anchor: 0aea7e5d-40d7-46c7-a8a6-479cb8061e49 | Idyllic Beachfront (fit) | This land enters tapped.
match: 16052b52-ade1-406f-a06b-ce7ea607fb63 | Stomping Ground (fit) | As this land enters, you may pay 2 life. If you don't, it enters tapped.
trap: 2acc1657-d716-41d7-aedf-9b080a58ec83 | Shambling Ghoul (fit) | This creature enters tapped.

## T028 · mechanism · accepted
anchor: 14589b6b-1814-46f9-a364-83cc15dacac2 | Diabolic Tutor (fit) | Search your library for a card, put that card into your hand, then shuffle.
match: ae18580a-1753-4dc8-b437-8a779b1a6e4d | Demonic Bargain (fit) | Exile the top thirteen cards of your library, then search your library for a card. Put that card into your hand, then shuffle.
trap: 299fc083-0834-4064-8344-f895aff68867 | Entomb (fit) | Search your library for a card, put that card into your graveyard, then shuffle.

## T029 · mechanism · accepted
anchor: 1c2ce0d0-8c3e-48e8-b1fd-54658bdb9af9 | Asphyxiate (fit) | Destroy target untapped creature.
match: 142af8f1-6d6b-4043-8cc4-6632481972c1 | Spread the Sickness (fit) | Destroy target creature, then proliferate.
trap: 1d001145-5d14-43a9-bf3b-3ce5c20b2a46 | Vengeance (fit) | Destroy target tapped creature.

## T030 · mechanism · accepted
anchor: 30e58e88-ac99-4d32-b2fe-6d6cb6288af3 | Magus of the Candelabra (fit) | {X}, {T}: Untap X target lands.
match: b57f06e4-da97-4473-9fbf-f66e9129502a | Argothian Elder (fit) | {T}: Untap two target lands.
trap: 6b1552f2-df64-485a-8509-b1916edf19cf | Mishra's Helix (fit) | {X}, {T}: Tap X target lands.

## T031 · mechanism · accepted
anchor: 86002526-ed3d-47e5-9f2a-55c4fa220ce8 | Folk of An-Havva (fit) | Whenever this creature blocks, it gets +2/+0 until end of turn.
match: 0b6625cb-4177-45bd-ba99-2af43dfd6f46 | Shu Defender (fit) | Whenever this creature blocks, it gets +0/+2 until end of turn.
trap: 0b5a3d7b-6b2c-4e91-8218-2b957277855c | Brazen Wolves (fit) | Whenever this creature attacks, it gets +2/+0 until end of turn.

## T032 · parameters · accepted
anchor: 013a465b-7034-45fe-bf62-c3d273131ec5 | Square Up (dev) | Target creature has base power and toughness 4/4 until end of turn.
match: 3e554ab4-8179-4d3e-84ba-9a67b51cff7f | Diminish (fit) | Target creature has base power and toughness 1/1 until end of turn.
trap: 3e4cb1b2-e2cc-4925-a226-6c6f1501d9c1 | Sorceress Queen (fit) | {T}: Target creature other than this creature has base power and toughness 0/2 until end of turn.

## T033 · parameters · accepted
anchor: 0931206b-eed2-40d0-9496-5ecefc7f8f90 | Infectious Horror (dev) | Whenever this creature attacks, each opponent loses 2 life.
match: 26b077df-7a05-459a-b5ff-bf27767bb519 | Vicious Conquistador (dev) | Whenever this creature attacks, each opponent loses 1 life.
trap: 8f933fae-6c0c-42d7-a817-14760d8285cd | Exquisite Blood (fit) | Whenever an opponent loses life, you gain that much life.

## T034 · parameters · accepted
anchor: 0c017406-7fc3-4701-93ec-ddb02044c12a | Piety (dev) | Blocking creatures get +0/+3 until end of turn.
match: 1ff55bbd-cde2-4fe9-a129-16790831d665 | Hold the Line (fit) | Blocking creatures get +7/+7 until end of turn.
trap: 62e3f285-886c-414e-b4ff-403a7c01c23a | Abbey Matron (fit) | {W}, {T}: This creature gets +0/+3 until end of turn.

## T035 · parameters · accepted
anchor: 110e5695-a9df-4201-9230-99f6abe9978f | Blade Sliver (dev) | All Sliver creatures get +1/+0.
match: 89d9641c-84a4-41df-af1f-5b552305e895 | Plated Sliver (fit) | All Sliver creatures get +0/+1.
trap: c7dd0614-e11e-4b6b-a2cb-9904e457148f | Hive Stirrings (fit) | Create two 1/1 colorless Sliver creature tokens.

## T036 · parameters · accepted
anchor: 120446cc-f55b-40db-915d-2669c1083e56 | Unnatural Predation (dev) | Target creature gets +1/+1 and gains trample until end of turn.
match: 0b3bdbc6-cde7-4187-b13c-82fd02ae4037 | Tread Upon (fit) | Target creature gets +2/+2 and gains trample until end of turn.
trap: 6881e3ac-9e3d-49d2-b2ac-63feb6f0f2ff | Relentless Hunter (fit) | {1}{R}{G}: This creature gets +1/+1 and gains trample until end of turn.

## T037 · parameters · accepted
anchor: 15e71226-feb0-4870-a3ea-d8604ca5b3dc | Puncturing Light (dev) | Destroy target attacking or blocking creature with power 3 or less.
match: 2bad07d2-6284-4f87-9e2e-2e9ec5d2dd57 | Searing Light (dev) | Destroy target attacking or blocking creature with power 2 or less.
trap: 1a852283-42d4-4197-91b9-5c952c7325ca | Kor Line-Slinger (fit) | {T}: Tap target creature with power 3 or less.

## T038 · parameters · accepted
anchor: 1f384f19-7fb4-4120-8001-5b401b4debbd | Reckless Pangolin (dev) | Whenever this creature attacks, it gets +1/+1 until end of turn.
match: 00895741-f94f-4d80-8ca4-c436bf3c8d56 | Steadfast Cathar (fit) | Whenever this creature attacks, it gets +0/+2 until end of turn.
trap: 2af1930c-a5f2-4e7e-910f-e84458b051ac | Deeproot Warrior (dev) | Whenever this creature becomes blocked, it gets +1/+1 until end of turn.

## T039 · parameters · accepted
anchor: 2afd64dd-94b3-4bc6-8411-465b3e07f3c9 | Emergent Growth (dev) | Target creature gets +5/+5 until end of turn and must be blocked this turn if able.
match: 256cc907-2a13-4f43-9d37-2e52034cdaf4 | Compelled Duel (fit) | Target creature gets +3/+3 until end of turn and must be blocked this turn if able.
trap: 9d277206-9156-4dc9-bdd2-9d9219787735 | Stonewood Invoker (fit) | {7}{G}: This creature gets +5/+5 until end of turn.

## T040 · parameters · accepted
anchor: 2cdd85c8-fa60-44ca-bc96-d38f029d2184 | Anurid Barkripper (dev) | Threshold — This creature gets +2/+2 as long as there are seven or more cards in your graveyard.
match: 042a665d-2f63-4f7e-ac89-0d560da68085 | Krosan Beast (fit) | Threshold — This creature gets +7/+7 as long as there are seven or more cards in your graveyard.
trap: ac5294c6-bc79-4a0f-b8b3-52b554561e38 | Frilled Cave-Wurm (fit) | Descend 4 — This creature gets +2/+0 as long as there are four or more permanent cards in your graveyard.

## T041 · parameters · accepted
anchor: 2f3bd1fc-387e-41b7-b19b-62dd9fbe2d10 | Thousand-legged Kami (dev) | Soulshift 7
match: 2efd1c86-41ba-4b86-bf63-e60e3263cb06 | Nightsoil Kami (fit) | Soulshift 5
trap: c6836d78-ca71-4413-aced-d4de19976221 | Birthing Boughs (fit) | {4}, {T}: Create a 2/2 colorless Shapeshifter creature token with changeling.

## T042 · parameters · accepted
anchor: 2f8c4805-694d-4408-8eb5-9fb7ce278ded | Springing Tiger (dev) | Threshold — This creature gets +2/+2 as long as there are seven or more cards in your graveyard.
match: 512477f0-1083-423b-9675-a8bc852f7d88 | Metamorphic Wurm (fit) | Threshold — This creature gets +4/+4 as long as there are seven or more cards in your graveyard.
trap: f5fa9138-9640-4586-bc81-caa6b8eae8b6 | Moorland Drifter (fit) | Delirium — This creature has flying as long as there are four or more card types among cards in your graveyard.

## T043 · parameters · accepted
anchor: 318f7f70-e374-40ef-8afb-3389c10461d8 | Syncopate (dev) | Counter target spell unless its controller pays {X}. If that spell is countered this way, exile it instead of putting it into its owner's graveyard.
match: ead1ee6a-e0da-46d5-89e8-31e8c0270bc2 | No More Lies (fit) | Counter target spell unless its controller pays {3}. If that spell is countered this way, exile it instead of putting it into its owner's graveyard.
trap: 626882c5-15a8-43ea-ab08-33a7e949c88c | Withdraw (fit) | Return target creature to its owner's hand. Then return another target creature to its owner's hand unless its controller pays {1}.

## T044 · parameters · accepted
anchor: 41011f09-ad8a-4eea-aa52-d20a219bc94c | Captivating Gyre (dev) | Return up to three target creatures to their owners' hands.
match: 4d5a970b-dd36-4871-a2aa-36ba21917cdf | Into the Void (fit) | Return up to two target creatures to their owners' hands.
trap: c30491b5-08fe-4694-9ac8-53be7b4b92ce | Portal of Sanctuary (fit) | {1}, {T}: Return target creature you control and each Aura attached to it to their owners' hands. Activate only during your turn.

## T045 · parameters · accepted
anchor: 44e9030d-dd49-46b4-a0eb-75a49b6638a2 | Maw of the Mire (dev) | Destroy target land. You gain 4 life.
match: 84e89aac-439c-4e09-b7d9-2ee047c224c5 | Drain the Well (dev) | Destroy target land. You gain 2 life.
trap: ee771e66-72f8-480f-9920-92c68ab93c3b | Ravenous Baloth (fit) | Sacrifice a Beast: You gain 4 life.

## T046 · parameters · accepted
anchor: 4e75fc29-68e3-41a9-9f59-adb4460663b2 | Fierce Invocation (dev) | Manifest the top card of your library, then put two +1/+1 counters on it.
match: d96ac790-428b-4a64-8dbd-6baa73eb6210 | Wildcall (fit) | Manifest the top card of your library, then put X +1/+1 counters on it.
trap: 0274f286-46a1-4028-878a-88c2146a895c | Sultai Emissary (fit) | When this creature dies, manifest the top card of your library.

## T047 · parameters · accepted
anchor: 509bfcbc-a59a-47c8-be78-29823fc010e3 | Ray of Dissolution (dev) | Destroy target enchantment. You gain 3 life.
match: ac5a13b1-0725-4978-8c93-cebd76be58ba | Cursebreak (fit) | Destroy target enchantment. You gain 2 life.
trap: 2dd3a6f7-65a9-4157-9ad9-13a885ebe927 | Bloodrite Invoker (fit) | {8}: Target player loses 3 life and you gain 3 life.

## T048 · parameters · accepted
anchor: 51abc95c-bd92-46f7-a0a1-6e9f8e0dda87 | Illness in the Ranks (dev) | Creature tokens get -1/-1.
match: adfba95e-c73c-415d-adea-edcd04372ba4 | Virulent Plague (fit) | Creature tokens get -2/-2.
trap: fb1755a0-3334-419b-8cb5-5a3ac7fa5b13 | Flowstone Surge (fit) | Creatures you control get +1/-1.

## T049 · parameters · accepted
anchor: 53e4273c-3292-47a0-90d1-8a0eb0eaf381 | Amphin Pathmage (dev) | {2}{U}: Target creature can't be blocked this turn.
match: 6eb21aca-116e-427b-9e75-8adaee0cc62a | Wormhole Serpent (fit) | {3}{U}: Target creature can't be blocked this turn.
trap: 2b43c7de-8c21-4137-b6b8-45965e4bb558 | Inkfathom Infiltrator (fit) | This creature can't block and can't be blocked.

## T050 · parameters · accepted
anchor: 576128ba-28aa-4ef9-9e72-8ca13f42c9b4 | Zephyr Charge (dev) | {1}{U}: Target creature gains flying until end of turn.
match: 98deac9e-28ea-4d11-8d75-833bf6d66e6e | Keen Glidemaster (fit) | {2}{U}: Target creature gains flying until end of turn.
trap: f610748a-8b2c-4c7d-b495-b4677ff4c423 | Kalastria Nightwatch (fit) | Whenever you gain life, this creature gains flying until end of turn.

## T051 · parameters · accepted
anchor: 58511cb8-348d-409e-8fd6-772be20e57cd | Silk Net (dev) | Target creature gets +1/+1 and gains reach until end of turn.
match: d96506b1-845d-4a9b-8414-456bb457d80f | Shape the Sands (fit) | Target creature gets +0/+5 and gains reach until end of turn.
trap: 66bd5999-742a-43e0-827c-ecb8ec7db242 | Ghost Warden (fit) | {T}: Target creature gets +1/+1 until end of turn.
